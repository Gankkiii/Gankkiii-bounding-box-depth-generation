import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
import random
import time 
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision
from convert_ckpt import add_additional_channels
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize, get_world_size
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
from inpaint_mask_func import draw_masks_from_boxes
from ldm.modules.attention import BasicTransformerBlock
from PIL import Image
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt
import gc


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling.
    type should be a list containing three values which sum should be 1

    It means the percentage of three stages:
    alpha=1 stage
    linear deacy stage
    alpha=0 stage.

    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.
    """
    if type == None:
        type = [1, 0, 0]

    assert len(type) == 3
    assert type[0] + type[1] + type[2] == 1

    stage0_length = int(type[0] * length)
    stage1_length = int(type[1] * length)
    stage2_length = length - stage0_length - stage1_length

    if stage1_length != 0:
        decay_alphas = np.arange(start=0, stop=1, step=1 / stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []

    alphas = [1] * stage0_length + decay_alphas + [0] * stage2_length

    assert len(alphas) == length

    return alphas

try:
    from apex import amp
except:
    pass


from depth_anything.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #



class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images, real, masked_real, captions, seen):
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        torchvision.utils.save_image( images, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_real.png')
        torchvision.utils.save_image( real, save_path, nrow=self.nrow)

        if masked_real is not None:
            # only inpaiting mode case 
            save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_mased_real.png')
            torchvision.utils.save_image( masked_real, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range)

        assert images.shape[0] == len(captions)

        save_path = os.path.join(self.base_path, 'captions.txt')
        with open(save_path, "a") as f:
            f.write( str(seen).zfill(8) + ':\n' )    
            for cap in captions:
                f.write( cap + '\n' )  
            f.write( '\n' ) 



def read_official_ckpt(ckpt_path):      
    "Read offical pretrained SD ckpt and convert into my style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
    return out 


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 






class Trainer:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )


        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)

        
        state_dict = read_official_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
        
        # modify the input conv for SD if necessary (grounding as unet input; inpaint)
        additional_channels = self.model.additional_channel_from_downsampler
        if self.config.inpaint_mode:
            additional_channels += 5 # 5 = 4(latent) + 1(mask)
        add_additional_channels(state_dict["model"], additional_channels)
        self.input_conv_train = True if additional_channels>0 else False

        # load original SD ckpt (with inuput conv may be modified) 
        missing_keys, unexpected_keys = self.model.load_state_dict( state_dict["model"], strict=False  )
        assert unexpected_keys == []
        original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 
        
        self.autoencoder.load_state_dict( state_dict["autoencoder"]  )
        self.text_encoder.load_state_dict( state_dict["text_encoder"]  )
        self.diffusion.load_state_dict( state_dict["diffusion"] , False )
 
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)

        # = = = = = = = = = = = = = load from ckpt: (usually for inpainting training) = = = = = = = = = = = = = #
        if self.config.ckpt is not None:
            first_stage_ckpt = torch.load(self.config.ckpt, map_location="cpu")
            first_stage_ckpt["model"].pop("position_net.linears.0.weight")
            #first_stage_ckpt.pop("position_net.null_positive_feature")
            #first_stage_ckpt.pop("position_net.linears.4.weight")
            #first_stage_ckpt.pop("position_net.linears.4.bias")
            self.model.load_state_dict(first_stage_ckpt["model"], strict=False)
            original_params_names = list(first_stage_ckpt["model"].keys())


        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = []
        trainable_names = []
        all_params_name = []
        for name, p in self.model.named_parameters():
            if "position_net" in name:
                params.append(p)
                trainable_names.append(name)
            else:
                assert name in original_params_names, name
            all_params_name.append(name)

            #if ("transformer_blocks" in name) and ("fuser" in name):
            #    # New added Attention layers
            #    params.append(p)
            #    trainable_names.append(name)
            #elif  "position_net" in name:
            #    # Grounding token processing network
            #    params.append(p)
            #    trainable_names.append(name)
            #elif  "downsample_net" in name:
            #    # Grounding downsample network (used in input)
            #    params.append(p)
            #    trainable_names.append(name)
            #elif (self.input_conv_train) and ("input_blocks.0.0.weight" in name):
            #    # First conv layer was modified, thus need to train
            #    params.append(p)
            #    trainable_names.append(name)
            #else:
            #    # Following make sure we do not miss any new params
            #    # all new added trainable params have to be haddled above
            #    # otherwise it will trigger the following error
            #    assert name in original_params_names, name
            #all_params_name.append(name)



        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(params)
        
        


        #  = = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters()) 
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()




        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False 




        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
        dataset_train = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=True, repeats=train_dataset_repeats)
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None 
        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=(sampler is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        if get_rank() == 0:
            total_image = dataset_train.total_images()
            print("Total training images: ", total_image)     
        



        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0  
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()


        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        
        # func return input for grounding tokenizer 
        self.grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input
        
        # func return input for grounding downsampler  
        self.grounding_downsampler_input = None
        if 'grounding_downsampler_input' in config:
            self.grounding_downsampler_input = instantiate_from_config(config.grounding_downsampler_input)

        if get_rank() == 0:       
            self.image_caption_saver = ImageCaptionSaver(self.name)

        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )





    @torch.no_grad()
    def get_input(self, batch):

        z = self.autoencoder.encode( batch["image"] )

        context = self.text_encoder.encode( batch["caption"]  )

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        
        inpainting_extra_input = None
        if self.config.inpaint_mode:
            # extra input for the inpainting model 
            inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
            masked_z = z*inpainting_mask
            inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
        
        grounding_extra_input = None
        if self.grounding_downsampler_input != None:
            grounding_extra_input = self.grounding_downsampler_input.prepare(batch)

        return z, t, context, inpainting_extra_input, grounding_extra_input 


    def run_one_step(self, batch, sampler,dep_model):
        alpha_generator_func = partial(alpha_generator, type=[0.3, 0.0, 0.7])
        sampler = DDIMSampler(self.diffusion, self.model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        x_start, t, context, inpainting_extra_input, grounding_extra_input = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        grounding_input = self.grounding_tokenizer_input.prepare(batch)
        input = dict(x=x_noisy, 
                    timesteps=t, 
                    context=context, 
                    inpainting_extra_input=inpainting_extra_input,
                    grounding_extra_input=grounding_extra_input,
                    grounding_input=grounding_input)
        model_output = self.model(input)

        consistent_t = torch.full_like(t, 999)
        x_noisy_consistent = self.diffusion.q_sample(x_start=x_start, t=consistent_t, noise=noise)
        input_consistent = dict(x=x_noisy_consistent,
                                timesteps=None,
                                context=context,
                                inpainting_extra_input=inpainting_extra_input,
                                grounding_extra_input=grounding_extra_input,
                                grounding_input=grounding_input)

        generate_image_start = time.time()
        result = sampler.sample(S=20, shape=(2, 4, 64, 64), input=input_consistent, uc=self.text_encoder.encode( len(batch["id"])*[""] ), guidance_scale=7.5,
                       mask=None, x0=None)
        del sampler
        gc.collect()
        torch.cuda.empty_cache()
        image_output_consistent = self.autoencoder.decode(result)
        get_depth_time_start = time.time()
        print(f"GTime:：{get_depth_time_start - generate_image_start}s")
        depth_consistent = self.get_depth(image_output_consistent,dep_model)
        get_depth_time_end = time.time()
        depth_tensor = torch.zeros_like(batch["depth_info"])
        print(f"DTime:：{get_depth_time_end  - get_depth_time_start}s")



         #Iterate over the depth_consistent and batch["boxes"]
        for i in range(len(depth_consistent)):
            for j in range(len(batch["boxes"][i])):
                    if batch["masks"][i][j] == 1:
                        depth_min, depth_max = depth_consistent[i].min(), depth_consistent[i].max()
                        depth_consistent_norm = (depth_consistent[i] - depth_min) * 255.0 / (depth_max - depth_min)

                    # Calculate the ROI coordinates
                        x_min, y_min, x_max, y_max = batch["boxes"][i][j]
                        x_min, y_min, x_max, y_max = int(x_min * 512), int(y_min * 512), int(x_max * 512), int(y_max * 512)

                    # Extract ROI and calculate average grayscale value
                        roi = depth_consistent_norm[y_min:y_max, x_min:x_max]
                        depth_tensor[i][j] = torch.mean(roi)




        # Save images for debugging (commented out)
        self.save_img(image_output_consistent[0], "image0.jpg", "c")
        self.save_img(image_output_consistent[1], "image1.jpg", "c")
        self.save_img(batch["image"][0], "image0_org.jpg", "c")
        self.save_img(batch["image"][1], "image1_org.jpg", "c")
        self.save_img(depth_consistent[0], "depth0.jpg", "l")
        self.save_img(depth_consistent[1], "depth1.jpg", "l")

        # Calculate the final loss

        orig_loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight
        depth_loss = torch.nn.functional.l1_loss(depth_tensor/255, batch["depth_info"]/255) * 0.01

        #loss = orig_loss + depth_loss
        #loss = orig_loss

        #self.loss_dict = {"loss": loss.item()}

        return orig_loss,depth_loss

    def save_img(self,image,name,m):
        if m =="l":
            image = image.cpu().detach().numpy()
            image = (image - image.min()) / (image.max() - image.min()) * 255.0
            image = Image.fromarray(image.astype(np.uint8), mode='L')
        else:
            image = torch.clamp(image, min=-1, max=1) * 0.5 + 0.5
            image = image.cpu().detach().numpy().transpose(1, 2, 0) * 255
            image = Image.fromarray(image.astype(np.uint8))
        image.save(os.path.join(r"D:\project\GLIGEN\generation_samples\test", name))



    def get_depth(self, image_tensor,depth_model):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        encoder = 'vitl'
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'D:/project/GLIGEN/depth_anything/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to(DEVICE).eval()
        depth = model.infer_image_tensor(image_tensor,'cuda')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return depth


    def start_training(self):
        alpha_generator_func = partial(alpha_generator, type=[0.3, 0.0, 0.7])
        sampler = DDIMSampler(self.diffusion, self.model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        encoder = 'vitl'
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'D:/project/GLIGEN/depth_anything/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to(DEVICE).eval()

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 )

        self.model.train()
        diag = dict()
        diag2 = dict()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx
            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            orig_loss, depth_loss = self.run_one_step(batch,sampler,model)
            loss = orig_loss + depth_loss
            self.loss_dict = {"loss": loss.item()}
            loss.backward()
            self.opt.step() 
            self.scheduler.step()

            gc.collect()
            torch.cuda.empty_cache()
            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)


            if (get_rank() == 0):
                if (iter_idx % 1 == 0):
                    self.log_loss()
                    diag[iter_idx] = orig_loss.cpu().detach().numpy()
                    diag2[iter_idx] = depth_loss.cpu().detach().numpy()
                    x_coords = list(diag.keys())
                    y_coords = list(diag.values())
                    y_coords2 = list(diag2.values())
                    plt.figure(figsize=(8, 6))
                    plt.plot(x_coords, y_coords, 'o-', label='orig Loss')
                    plt.savefig('orig_loss.png')
                    plt.figure(figsize=(8, 6))
                    plt.plot(x_coords, y_coords2, 'o-', label='depth Loss')
                    plt.savefig('depth_loss.png')
                    plt.close('all')
                if (iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 )  or  (iter_idx == self.config.total_iters-1):
                    self.save_ckpt_and_result()
            synchronize()

        
        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    

    @torch.no_grad()
    def save_ckpt_and_result(self):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch 
            batch_here = self.config.batch_size
            batch = sub_batch( next(self.loader_train), batch_here)
            batch_to_device(batch, self.device)

            
            if "boxes" in batch:
                real_images_with_box_drawing = [] # we save this durining trianing for better visualization
                for i in range(batch_here):
                    temp_data = {"image": batch["image"][i], "boxes":batch["boxes"][i]}
                    im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                    real_images_with_box_drawing.append(im)
                real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
            else:
                # keypoint case 
                real_images_with_box_drawing = batch["image"]*0.5 + 0.5 
                
            
            uc = self.text_encoder.encode( batch_here*[""] )
            context = self.text_encoder.encode(  batch["caption"]  )
            
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)      
            shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)
            
            # extra input for inpainting 
            inpainting_extra_input = None
            if self.config.inpaint_mode:
                z = self.autoencoder.encode( batch["image"] )
                inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
                masked_z = z*inpainting_mask
                inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)
            
            grounding_extra_input = None
            if self.grounding_downsampler_input != None:
                grounding_extra_input = self.grounding_downsampler_input.prepare(batch)
            
            grounding_input = self.grounding_tokenizer_input.prepare(batch)
            input = dict( x=None, 
                          timesteps=None, 
                          context=context, 
                          inpainting_extra_input=inpainting_extra_input,
                          grounding_extra_input=grounding_extra_input,
                          grounding_input=grounding_input )
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)
            
            autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            masked_real_image =  batch["image"]*torch.nn.functional.interpolate(inpainting_mask, size=(512, 512)) if self.config.inpaint_mode else None
            self.image_caption_saver(samples, real_images_with_box_drawing,  masked_real_image, batch["caption"], iter_name)

        ckpt = dict(model = model_wo_wrapper.state_dict(),
                    text_encoder = self.text_encoder.state_dict(),
                    autoencoder = self.autoencoder.state_dict(),
                    diffusion = self.diffusion.state_dict(),
                    opt = self.opt.state_dict(),
                    scheduler= self.scheduler.state_dict(),
                    iters = self.iter_idx+1,
                    config_dict=self.config_dict,
        )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        torch.save( ckpt, os.path.join(self.name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(self.name, "checkpoint_latest.pth") )



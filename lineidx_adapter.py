def find_line_byte_offsets(file_path):
    offsets = []
    current_offset = 0

    with open(file_path, 'rb') as f:
        for line in f:
            offsets.append(current_offset)
            current_offset += len(line)
    
    return offsets

def save_offsets_to_file(offsets, output_file_path):
    with open(output_file_path, 'w') as f:
        for offset in offsets:
            f.write(f'{offset}\n')

def main():
    input_file_path = r'D:\project\GLIGEN\DATA\dataset\train-00.tsv'  # 请将此路径替换为你的TSV文件路径
    output_file_path = r'D:\project\GLIGEN\DATA\dataset\train-00.lineidx'  # 输出文件的路径

    offsets = find_line_byte_offsets(input_file_path)
    save_offsets_to_file(offsets, output_file_path)
    print(f'Offsets have been saved to {output_file_path}')

if __name__ == "__main__":
    main()




def load_parallel_texts(russian_file_path, english_file_path):
    parallel_texts = []
    
    # Open both files and view them side by side
    with open(russian_file_path, 'r', encoding='utf-8') as ru_file, open(english_file_path, 'r', encoding='utf-8') as en_file:
        for ru_line, en_line in zip(ru_file, en_file):
            # Strip lines to remove leading/trailing whitespace and pair them
            ru_text = ru_line.strip()
            en_text = en_line.strip()
            parallel_texts.append((ru_text, en_text))
    
    return parallel_texts



def write_to_file(data, filename):
    f = open(filename, "a")
    for pair in data:
        f.write(pair[0])
        f.write('\n')
        f.write(pair[1])
        f.write('\n')
    f.close()



def load_one_file_texts(parallel_text_path):
    with open(parallel_text_path, 'r') as f:
        lines = f.readlines()

    # Remove any trailing newline characters and other whitespace
    lines = [line.strip() for line in lines]

    # Pairing lines into tuples (data[0], data[1])
    return [(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]


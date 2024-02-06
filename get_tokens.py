def get_tokens(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    content = "\n".join(lines)
    tokens = content.split()

    return tokens

from config import opt

def preview_args(args):
    print('========== Begin something about args')

    print('Hidden Size:', args.hidden_dim)
    print('Text Dim:', args.embedding_dim)
    print()
    print('Epoch:', args.epoch)
    print('Device:', opt.device)
    print('Device ID:', args.device_id)
    print()
    print('Batch Size', args.batch_size)
    print("Attention", args.attention)
    print("Content", args.content)
    print("Reply", args.reply)
    print("Bert vector", args.bert_vector)
    if args.bert_vector == True:
        print("Bert vector type", args.bert_vector_type)
    print('========== End something about args')


if __name__ == '__main__':
    preview_args()

import torch
import argparse
import os
home = os.getenv("HOME")

class Config(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "./text_data"

    weibo_dataset = os.path.join(data_path, 'weibo', 'dataset')
    label_weibo = os.path.join(data_path, 'weibo', 'label', 'weibo.txt')

    embedding_path = './embedding_data'

    bert_embedding_path = os.path.join(embedding_path, 'bert')
    weibo_bert_vector = os.path.join(bert_embedding_path, 'weibo_bert.tar')
    weibo_sentiment_vector = os.path.join(bert_embedding_path, 'weibo_sentiment.tar')

    tencent_embedding_path = os.path.join(embedding_path, 'tencent' ,'Tencent_AILab_ChineseEmbedding.txt')
    tencent_embedding_dim = 200

    max_document_len = 60
    max_reply_len = 20
    max_user_description_len = 20
    max_verified_reason_len = 8


opt = Config()

parser = argparse.ArgumentParser(description='JARVIS')
## learning
parser.add_argument('-epoch', type=int, default=60, help='number of epochs for train [default: 60]')
parser.add_argument('-random_seed', type=int, default=123456, help='random_seed [default: 123456]')
parser.add_argument('-batch-size', type=int, default=1, help='batch size for training [default: 1]')
parser.add_argument('-hidden-dim', type=int, default=300, help='number of hidden dimension [default: 256]')
parser.add_argument('-device_id', type=int, default=0, help='device-id [default: 0]')
## view
parser.add_argument('-corpus', type=str, default="weibo", help='train corpus [default: weibo]')
parser.add_argument('-content', action='store_true', default=False, help='content view switch [default: False]')
parser.add_argument('-reply', action='store_true', default=False, help='reply view switch [default: False]')
parser.add_argument('-bert_vector', action='store_true', default=False, help='sentiment view switch [default: False]')
parser.add_argument('-bert_vector_type', type=str, default="sentiment", help='sentiment view type sentiment/bert [default: sentiment]')
parser.add_argument('-attention', action='store_true', default=False, help='attention switch for reply and sentiment views [default: False]')
parser.add_argument('-content_attention', action='store_true', default=True, help='content attention switch [default: True]')
parser.add_argument('-content_length_limit', type=int, default=-1, help='-1:without limit')
parser.add_argument('-reply_number_limit', type=int, default=4096, help='reply num limit [default: 4096]')
parser.add_argument('-reply_sample_frequency', type=int, default=1, help='1:without sample')

args = parser.parse_args()

args.multi_view_num = int(args.content) + int(args.reply) + int(args.bert_vector)
args.multi_view_name = ''
if args.content:
    args.multi_view_name += 'content-'
if args.reply:
    args.multi_view_name += 'reply-'
if args.bert_vector:
    args.multi_view_name += args.bert_vector_type + '-'
if args.attention:
    args.multi_view_name += 'attention-'

args.multi_view_name += 'content-' + str(args.content_length_limit) + '-'
args.multi_view_name += 'reply-' + str(args.reply_number_limit) + '-'
args.multi_view_name += 'sample-' + str(args.content_length_limit) + '-'
args.multi_view_name += 'hidden_dim-' + str(args.hidden_dim)

if args.corpus == 'weibo':
    args.corpus_path = opt.weibo_dataset

    if args.bert_vector_type == 'bert':
        args.bert_vector_path = opt.weibo_bert_vector
    elif args.bert_vector_type == 'sentiment':
        args.bert_vector_path = opt.weibo_sentiment_vector
    else:
        print('bert_vector_type error, plsease check.')
        exit()

    args.save_dir = os.path.join(opt.data_path, 'weibo', 'result')
    args.embedding_path = opt.tencent_embedding_path
    args.embedding_dim = opt.tencent_embedding_dim

else:
    print('corpus error, plsease check.')
    exit()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if opt.device == torch.device('cuda'):
    torch.cuda.set_device(args.device_id)


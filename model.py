from torch import nn
import torch
from attention import Attention
from config import opt


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.content = nn.GRU(input_size=self.args.embedding_dim, hidden_size=self.args.hidden_dim, bidirectional=False, num_layers=1).to(opt.device)
        self.reply = nn.GRU(input_size=self.args.embedding_dim, hidden_size=self.args.hidden_dim, bidirectional=False, num_layers=1).to(opt.device)
        self.bert_vector = nn.GRU(input_size=768, hidden_size=self.args.hidden_dim, bidirectional=False, num_layers=1).to(opt.device)
        self.linear = nn.Linear(self.args.hidden_dim, self.args.hidden_dim).to(opt.device)  # output: (L, B, H) -> (L, B, H)
        self.content_linear = nn.Linear(self.args.hidden_dim, self.args.hidden_dim).to(opt.device)  # output: (L, B, H) -> (L, B, H)
        self.output_linear = nn.Linear(self.args.multi_view_num * self.args.batch_size * self.args.hidden_dim, 1).to(opt.device)
        self.attention = Attention(self.args)

    def forward(self, event):
        multi_view = torch.tensor([]).to(opt.device)

        # content
        if self.args.content:
            #print(self.args.multi_view_name)
            #print(event.content)
            X_content_data = torch.tensor(event.content_word_embedding_array, dtype=torch.float).to(opt.device)
            X_content_data = X_content_data.unsqueeze(1).to(opt.device)

            output_content, hidden_content = self.content(X_content_data)
            output_content = self.content_linear(output_content)
            if self.args.content_attention:
                output_content, attn_dist_content = self.attention(output_content)
            else:
                output_content = output_content.sum(0)/output_content.size(0)
            output_content = output_content.flatten()
            multi_view = torch.cat((multi_view, output_content)).to(opt.device)

        # reply
        if self.args.reply:
            #print(event.reply_no_same)
            X_reply_data = torch.tensor(event.reply_sentence_embedding_array, dtype=torch.float).to(opt.device)
            X_reply_data = X_reply_data.unsqueeze(1).to(opt.device)

            output_reply, hidden_reply = self.reply(X_reply_data)
            output_reply = self.linear(output_reply)
            if self.args.attention:
                output_reply, atten_dist_reply = self.attention(output_reply)
            else:
                output_reply = output_reply.sum(0)/output_reply.size(0)
            output_reply = output_reply.flatten()
            multi_view = torch.cat((multi_view, output_reply)).to(opt.device)

        # bert vector
        if self.args.bert_vector:
            X_bert_vector_data = torch.tensor(event.bert_vector, dtype=torch.float).to(opt.device)

            output_bert_vector,hidden_bert_vector = self.bert_vector(X_bert_vector_data)
            output_bert_vector = self.linear(output_bert_vector)
            if self.args.attention:
                output_bert_vector, atten_dist_bert_vector = self.attention(output_bert_vector)
            else:
                output_bert_vector = output_bert_vector.sum(0)/output_bert_vector.size(0)
            output_bert_vector = output_bert_vector.flatten()
            multi_view = torch.cat((multi_view, output_bert_vector)).to(opt.device)

        result = torch.sigmoid(self.output_linear(multi_view))
        loss = (event.label - result)**2
        return loss, result


if __name__ == '__main__':
    model = Model()
    batch = torch.tensor([[0, 1]])
    print(batch.shape)
    print(model.embed(batch).shape)

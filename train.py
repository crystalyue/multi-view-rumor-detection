from torch import optim
import time
import os
import torch
import glob

from dataset import Dataset
from embedding import build_embedding_matrix
from model import Model
from config import opt, args
from preview import preview_args
from vocab import build_vocab


class Train():
    def __init__(self, arg, load_model=False):
        self.args = arg
        self.vocab = build_vocab(self.args)
        self.embedding_matrix = build_embedding_matrix(self.args, self.vocab)

        self.dataset_train = Dataset(self.args, 'train', self.embedding_matrix)
        self.dataset_val = Dataset(self.args, 'val', self.embedding_matrix)
        self.dataset_test = Dataset(self.args, 'test', self.embedding_matrix)

        self.model = Model(self.args)
        self.optimizer = optim.Adam(self.model.parameters())
        self._epoch = 0
        self._iter = 0
        self.max_val_acc = None
        self.max_test_acc = None
        if load_model:
            self.load_model()

    def save_model(self):
        state = {
            'epoch': self._epoch,
            'iter': self._iter,
            'max_val_acc': self.max_val_acc,
            'max_test_acc': self.max_test_acc,
            'state_dict': self.model.state_dict(),
        }

        name = '{}.pth'.format(self._epoch)
        multi_view_dir = os.path.join(self.args.save_dir, self.args.multi_view_name)
        if not os.path.exists(multi_view_dir):
            os.mkdir(multi_view_dir)
        model_save_dir = os.path.join(multi_view_dir, 'model')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        model_save_path = os.path.join(model_save_dir, name)
        print("saving model", model_save_path)
        torch.save(state, model_save_path)

    def load_model(self):
        assert os.path.exists(self.args.save_dir)

        if len(glob.glob(os.path.join(self.args.save_dir, self.args.multi_view_name, 'model', '*.pth'))) == 0:
            return

        f_list = glob.glob(os.path.join(self.args.save_dir, self.args.multi_view_name, 'model', '*.pth'))
        epoch_list = [int(os.path.split(i)[1].split('.')[0]) for i in f_list]
        start_epoch = sorted(epoch_list)[-1]
        name = '{}.pth'.format(start_epoch)
        file_path = os.path.join(self.args.save_dir, self.args.multi_view_name, 'model', name)
        print("loading model", file_path)

        if opt.device == torch.device('cuda'):
            state = torch.load(file_path)
        else:
            state = torch.load(file_path, map_location=opt.device)

        self._epoch = state['epoch']
        self._iter = state['iter']
        self.max_val_acc = state['max_val_acc']
        self.max_test_acc = state['max_test_acc']
        print('max_val_acc:',self.max_val_acc)
        print('max_test_acc:', self.max_test_acc)

        self.model.load_state_dict(state['state_dict'])

    def save_result(self, val_acc, test_acc):
        multi_view_dir = os.path.join(self.args.save_dir, self.args.multi_view_name)
        if not os.path.exists(multi_view_dir):
            os.mkdir(multi_view_dir)
        result_file_dir = os.path.join(multi_view_dir, 'result')
        if not os.path.exists(result_file_dir):
            os.mkdir(result_file_dir)
        result_file_path = os.path.join(result_file_dir, 'result.txt')
        with open(result_file_path, 'a') as fp:
            fp.write('time:' + str(time.time()) + ' epoch:' + str(self._epoch) + ' iter:' + str(self._iter) + ' val_acc:' + str(val_acc) + ' test_acc:' + str(test_acc) + '\n')
            fp.flush()

    def save_confuse_info(self, TP, FP, TN, FN):
        multi_view_dir = os.path.join(self.args.save_dir, self.args.multi_view_name)
        if not os.path.exists(multi_view_dir):
            os.mkdir(multi_view_dir)
        confuse_info_dir = os.path.join(multi_view_dir, 'confuse_info')
        if not os.path.exists(confuse_info_dir):
            os.mkdir(confuse_info_dir)
        confuse_info_save_path = os.path.join(confuse_info_dir, str(self._epoch) + '.txt')
        with open(confuse_info_save_path, 'w') as fp_confuse:
            fp_confuse.write(str(TP) + '\n')
            fp_confuse.write(str(FP) + '\n')
            fp_confuse.write(str(TN) + '\n')
            fp_confuse.write(str(FN) + '\n')

    def train(self):
        train_events = self.dataset_train.events()
        val_events = self.dataset_val.events()
        test_events = self.dataset_test.events()

        multi_view_dir = os.path.join(self.args.save_dir, self.args.multi_view_name)
        if not os.path.exists(multi_view_dir):
            os.mkdir(multi_view_dir)
        multi_view_result_dir = os.path.join(multi_view_dir, 'result')
        if not os.path.exists(multi_view_result_dir):
            os.mkdir(multi_view_result_dir)
        predict_result_path = os.path.join(multi_view_result_dir, 'test_predict.txt')
        fp_predict = open(predict_result_path, 'w')

        for i in range(0, self.args.epoch):
            start_time = time.time()
            TP = []
            FP = []
            TN = []
            FN = []
            for train_event in train_events:
                if len(train_event.content) == 0:
                    continue
                self._iter += 1
                self.optimizer.zero_grad()
                loss, result = self.model(train_event)

                if result >= 0.5:
                    result = 1
                else:
                    result = 0

                if result == 1 and train_event.label == 1:
                    TP.append(train_event.filename)
                elif result == 1 and train_event.label == 0:
                    FP.append(train_event.filename)
                elif result == 0 and train_event.label == 0:
                    TN.append(train_event.filename)
                elif result == 0 and train_event.label == 1:
                    FN.append(train_event.filename)

                print('loss:', loss, 'epoch:', self._epoch, 'iter:', self._iter,
                      'time:', time.time() - start_time, "line length:", train_event.len_reply)
                start_time = time.time()
                loss.backward()
                self.optimizer.step()

            # save train_confuse_info and model
            self.save_confuse_info(TP, FP, TN, FN)

            # val
            val_event_num = 0
            val_true_num = 0

            for val_event in val_events:
                if len(val_event.content) == 0:
                    continue
                val_event_num += 1
                # print(val_event_num)
                val_loss, val_result= self.model(val_event)
                print(val_loss)
                if val_result >= 0.5:
                    val_result = 1
                else:
                    val_result = 0
                if val_result == val_event.label:
                    val_true_num += 1

            self._epoch += 1
            val_acc = val_true_num/val_event_num
            print(val_acc)

            if self.max_val_acc is None:
                self.max_val_acc = val_acc
            elif self.max_val_acc < val_acc:
                self.max_val_acc = val_acc

            self.save_model()

            # test
            test_event_num = 0
            test_true_num = 0
            for test_event in test_events:
                if len(test_event.content) == 0:
                    continue
                test_event_num += 1

                test_loss, test_result = self.model(test_event)

                print(test_loss)
                if test_result >= 0.5:
                    test_result = 1
                else:
                    test_result = 0
                if test_result == test_event.label:
                    test_true_num += 1

                fp_predict.write(
                    'epoch:' + str(self._epoch) + ' event:' + test_event.filename + ' label:' + str(
                        test_event.label) + ' predict:' + str(
                        test_result) + '\n')

            test_acc = test_true_num / test_event_num

            self.save_result(val_acc, test_acc)


    def test(self):
        model_dir = os.path.join(self.args.save_dir, self.args.multi_view_name, 'model')
        result_dir = os.path.join(self.args.save_dir, self.args.multi_view_name, 'result')
        val_result_path = os.path.join(result_dir, 'result.txt')
        test_result_path = os.path.join(result_dir, 'test_result.txt')
        predict_result_path = os.path.join(result_dir, 'test_predict.txt')
        fp_predict = open(predict_result_path, 'w')
        # read val result from file
        val_result = []
        with open(val_result_path) as fp_val:
            lines = fp_val.readlines()
            for line in lines:
                if line == '':
                    continue
                line_splits = line.split(' ')
                epoch = line_splits[1].split(':')[1]
                val_acc = float(line_splits[3].split(':')[1])
                val_result.append([epoch, val_acc])

        # sort val result
        val_result.sort(key=lambda val: val[1], reverse=True)

        for i in range(0,3):
            epoch_str = val_result[i][0]
            val_acc = val_result[i][1]

            model_file_path = os.path.join(model_dir, epoch_str + '.pth')
            print("loading model", model_file_path)
            if opt.device == torch.device('cuda'):
                state = torch.load(model_file_path)
            else:
                state = torch.load(model_file_path, map_location=opt.device)
            self.model.load_state_dict(state['state_dict'])

            # test
            event_num = 0
            true_num = 0
            test_events = self.dataset_test.events()
            for event in test_events:
                if len(event.content) == 0:
                    continue
                event_num += 1

                loss, result = self.model(event)

                print(loss)
                if result >= 0.5:
                    result = 1
                else:
                    result = 0
                if result == event.label:
                    true_num += 1

                fp_predict.write('epoch:' + epoch_str + ' event:' + event.filename + ' label:' + str(event.label) + ' predict:' + str(result) + '\n')

            fp_predict.write('\n\n')

if __name__ == '__main__':

    preview_args(args)
    train = Train(args, load_model=False)
    train.train()
    #train.test()
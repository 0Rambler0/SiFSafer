import os 
from concurrent.futures import ThreadPoolExecutor

def exec(cmd):
    os.system(cmd)

if __name__ == '__main__':
    dataset = 'ASVspoof2019 or ASVspoof2021'
    data_path = "raw data path"
    save_data_path = "save data path"
    subset = ['train', 'dev', 'eval'] if dataset == "ASVspoof2019" else ['LA', 'DF']
    for set in subset:
        if dataset == "ASVspoof2021":
            dataset_path = '{}/ASVspoof2021_{}_eval/flac'.format(data_path, set)
            save_path = '{}/ASVspoof/ASVspoof2021_{}_eval_silence_new/flac'.format(save_data_path, set)
        else:
            dataset_path = '{}/ASVspoof2019_LA_{}/flac'.format(data_path, set)
            save_path = '{}/ASVspoof2019_LA_{}_silence_new/flac'.format(save_data_path, set)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        pool = ThreadPoolExecutor(24)
        fname_list = os.listdir(dataset_path)
        total = len(fname_list)
        count = 0
        for fname in fname_list:
            count += 1
            fname = fname.split('.')[0]
            target_path = os.path.join(dataset_path, fname)
            target_save_path = os.path.join(save_path, fname)
            cmd3 = 'sox {}.flac {}.flac silence 1 0.00001 1% -1 0.00001 1%%'.format(target_path, target_save_path)
            pool.submit(exec, cmd3)
            # os.system(cmd3)
            if count%100 == 0:
                print('processed rate: {}/{}'.format(count, total), end='\r', flush=False)
        cmd4 = 'rm *.wav'
        os.system(cmd4)
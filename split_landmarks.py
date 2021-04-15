import json
import os

def save_batches(landmarks):
    batch_size = int(len(landmarks) / _get_num_of_threads())
    filename = 'run_bi_online_generation.sh'
    f = open(filename, 'w')
    for i in range(_get_num_of_threads()):
        batch = dict(list(landmarks.items())[i * batch_size: (i + 1) * batch_size])
        save_name = "landmarks/landmarks_"+str(i)+".json"
        with open(save_name, 'w') as fp:
            json.dump(batch, fp)
            f.write("nohup python bi_online_generation.py --path " +str(save_name)+"&> logs/logs"+str(i)+".logs &"+"\n")
        print("Completed chunk "+str(i))
    f.close()
    print("Process completed \nDo sh "+filename)

def _get_num_of_threads() -> int:
    return os.cpu_count()

if __name__ == '__main__':
    landmarks_path = 'landmarks.json'
    with open(landmarks_path, 'r') as f:
        landmarks_json = json.load(f)
    save_batches(landmarks_json)




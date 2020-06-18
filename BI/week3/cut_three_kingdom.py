import jieba
import os

souce_file = './three_kingdoms/source'
segment_file = './three_kingdoms/segment'

file_name = './three_kingdoms/source/three_kingdoms.txt'

def seg_file(file_name, out_file_dir, stop_word=[]):
    out_file_name = out_file_dir + '/seg_threekingdoms.txt'
    print(out_file_name)
    with open(file_name, 'rb') as f:
        decument = f.read()
        # print(decument)
        decument_cut = jieba.cut(decument)
        # print(decument_cut)
        senstence_seg = []
        for word in decument_cut:
            if word not in stop_word:
                senstence_seg.append(word)
                # print(senstence_seg)
            result = ' '.join(senstence_seg)
            result = result.encode('utf-8')
            # print(result)
            with open(out_file_name, 'wb') as f2:
                f2.write(result)
        print('over')
seg_file(file_name, segment_file)

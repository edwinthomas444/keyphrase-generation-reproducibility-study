import sys
from nltk.stem.porter import PorterStemmer
import argparse

stemmer = PorterStemmer()


def norm(container):
    '''消除每一个元素的空格，方便比较'''
    try:
        remove_space_set = set(["".join(w.split(" ")) for w in container])
        return remove_space_set
    except AttributeError:
        print(container)
        print("AttributeError")


def stem_norm(container):
    '''消除每一个元素的空格，方便比较，同时做stem处理'''
    result_set = set()
    remove_space_set = set([w.strip() for w in container])
    for keyphrase in remove_space_set:
        stem_keyphrase_list = []
        for word in keyphrase.split(" "):
            stem_keyphrase_list.append(stemmer.stem(word))
        result_set.add(" ".join(stem_keyphrase_list))
    return result_set


def restore_article(input_words):
    article = []
    cur_word = ""
    for w in input_words:
        if w[:2] == '##':
            cur_word += w[2:]
        elif len(cur_word) > 0:
            article.append(cur_word)
            cur_word = w
        else:
            cur_word = w
    if len(cur_word) > 0:
        article.append(cur_word)
    return " " + " ".join(article) + " "


def process(input_file, preds_file, golds_file, stem_divide, stem_evaluate):
    doc_list = []
    golds_abs_list = []
    preds_list = []
    preds_list_top5 = []
    preds_list_topo = []

    fp = open(preds_file)
    fg = open(golds_file)

    fp = fp.readlines()
    fg = fg.readlines()

    with open(input_file, "r") as fi:
        for line in fi:
            doc_list.append(restore_article(line.strip().split(" ")))

    for line, line_g in zip(fp, fg):
        # find ground truth total kps
        gt_parts = line_g.strip().split(' ; ')
        o = len(gt_parts)

        parts = line.strip().split(' ; ')
        k_set = set()
        k_set_top5 = set()
        k_set_topo = set()

        # for i in parts:
        for idx, i in enumerate(parts):
            k_set.add(i)
            if idx < 5:
                k_set_top5.add(i)
            if idx < o:
                k_set_topo.add(i)

            #k_set.add(i.replace(' ', ''))
        if stem_evaluate:
            k_set = stem_norm(k_set)
            k_set_top5 = stem_norm(k_set_top5)
            k_set_topo = stem_norm(k_set_topo)
        else:
            k_set = norm(k_set)
            k_set_top5 = norm(k_set_top5)
            k_set_topo = norm(k_set_topo)

        preds_list.append(k_set)
        preds_list_top5.append(k_set_top5)
        preds_list_topo.append(k_set_topo)

    if stem_divide:
        for line in fg:
            parts = line.strip().split(' ; ')
            k_abs_set = set()
            for i in parts:
                k_abs_set.add(i.replace(' ##', ''))
                # k_abs_set.add(i.replace(' ##', '').replace(' ',''))
            if stem_evaluate:
                k_abs_set = stem_norm(k_abs_set)
            else:
                k_abs_set = norm(k_abs_set)
            golds_abs_list.append(k_abs_set)
    else:
        for idx, line in enumerate(fg):
            all_keyphrase = set(line.strip().split('@'))
            k_abs_set = set()
            for keyphrase in all_keyphrase:
                if ' ' + keyphrase + ' ' not in doc_list[idx]:
                    k_abs_set.add(keyphrase)
            if stem_evaluate:
                k_abs_set = stem_norm(k_abs_set)
            else:
                k_abs_set = norm(k_abs_set)
            golds_abs_list.append(k_abs_set)

    # print(golds_abs_list)
    return preds_list, preds_list_top5, preds_list_topo, golds_abs_list


def evaluate(golds_list, preds_list, top5=False, topo=False, display=True):
    p = 0
    r = 0
    corr_num_p = 0
    sample_num_p = 0
    corr_num_r = 0
    sample_num_r = 0
    count_sample = 0
    assert len(golds_list) == len(
        preds_list), f'{len(golds_list)}!={len(preds_list)}'
    for i in range(len(golds_list)):
        count_sample += 1
        for it in preds_list[i]:
            if it == 'none':
                continue
            if it in golds_list[i]:
                corr_num_p += 1
            sample_num_p += 1
        for it in golds_list[i]:
            if it == 'none':
                continue
            if it in preds_list[i]:
                corr_num_r += 1
            sample_num_r += 1

    eps = 1e-08
    p = 1.0 * corr_num_p / (sample_num_p + eps)
    r = 1.0 * corr_num_r / (sample_num_r + eps)
    f = 2 * p * r / (p + r + eps)

    if not display:
        return p, r, f

    if sample_num_p == 0 or sample_num_r == 0:
        if top5:
            print("micro absent keyphrase p,r,f@5")
        elif topo:
            print("micro absent keyphrase p,r,f@o")
        else:
            print("micro absent keyphrase p,r,f@m")
        print("p : 0.0  r : 0.0  f : 0.0")

    if top5:
        print("micro absent keyphrase p,r,f@5")
    elif topo:
        print("micro absent keyphrase p,r,f@o")
    else:
        print("micro absent keyphrase p,r,f@m")

    print(round(p, 4), round(r, 4), round(f, 4))
    # print("average predicted absent keyphrases %f" % (sample_num_p / count_sample))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--stem_evaluate', action='store_true', default=False,
                        help="whether use stem before evaluation, default: False")
    parser.add_argument('--stem_divide', action='store_true', default=False,
                        help="whether use stem before dividing keyphrases into present and absent parts, default: False")
    parser.add_argument(
        '--input_path', help='the path to the input .seq.in file')
    parser.add_argument(
        '--pred_path', help='the path to the predict model.absent file')
    parser.add_argument(
        '--gold_path', help='the path to the ground truth file, if stem_divide, it is .absent file; else, it is .keyphrase file')
    parser.add_argument(
        '--output_path', help='output path where for metric significance computation f-score values')
    args = parser.parse_args()
    # print(args)
    path_words = args.input_path
    path_p = args.pred_path
    path_gold = args.gold_path
    stem_evaluate = args.stem_evaluate
    stem_divide = args.stem_divide

    if (".absent" in path_gold and stem_divide) or (".keyphrase" in path_gold and not stem_divide):
        pass
    else:
        sys.exit("Error, when stem_divide we use .absent(which is provided by meng) as ground truth; when not stem_divide we use .keyphrase(contain all keyphrases and we divide by ourselves) as ground truth. please check your settings")

    preds_list, preds_list_top5, preds_list_topo, golds_abs_list = process(
        path_words, path_p, path_gold, stem_divide, stem_evaluate)

    # micro scores
    evaluate(golds_abs_list, preds_list)
    evaluate(golds_abs_list, preds_list_top5, top5=True)
    evaluate(golds_abs_list, preds_list_topo, topo=True)

    ###### MACRO SCORES ##########
    macro_fm_list = []
    macro_fk_list = []
    macro_fo_list = []

    # top-m
    eps = 1e-08
    macro_p, macro_r, macro_f = 0, 0, 0
    count = 0
    for g, p in zip(golds_abs_list, preds_list):
        p, r, f = evaluate([g], [p], display=False)
        macro_p += p
        macro_r += r
        count += 1
        macro_fm_list.append(f)

    macro_p = macro_p/(count+eps)
    macro_r = macro_r/(count+eps)
    macro_f = 2 * macro_p * macro_r / (macro_r + macro_p + eps)
    print("macro absent keyphrase p,r,f@m")
    print(round(macro_p, 4), round(macro_r, 4), round(macro_f, 4))

    ## top-k (5)
    eps = 1e-08
    macro_p, macro_r, macro_f = 0, 0, 0
    count = 0
    for g, p in zip(golds_abs_list, preds_list_top5):
        p, r, f = evaluate([g], [p], top5=True, display=False)
        macro_p += p
        macro_r += r
        count += 1
        macro_fk_list.append(f)

    macro_p = macro_p/(count+eps)
    macro_r = macro_r/(count+eps)
    macro_f = 2 * macro_p * macro_r / (macro_r + macro_p + eps)
    print("macro absent keyphrase p,r,f@5")
    print(round(macro_p, 4), round(macro_r, 4), round(macro_f, 4))

    ## top-k (o)
    eps = 1e-08
    macro_p, macro_r, macro_f = 0, 0, 0
    count = 0
    for g, p in zip(golds_abs_list, preds_list_topo):
        p, r, f = evaluate([g], [p], topo=True, display=False)
        macro_p += p
        macro_r += r
        count += 1
        macro_fo_list.append(f)

    macro_p = macro_p/(count+eps)
    macro_r = macro_r/(count+eps)
    macro_f = 2 * macro_p * macro_r / (macro_r + macro_p + eps)
    print("macro absent keyphrase p,r,f@o")
    print(round(macro_p, 4), round(macro_r, 4), round(macro_f, 4))

    # save macro per sample f scores for significance testing
    import os
    op_path = args.output_path
    types = ['m','k','o']
    for type in types:
        fpath = os.path.join(op_path, f'significance_absent_macro_f_{type}')
        with open(fpath, 'w') as f:
            for score in eval(f'macro_f{type}_list'):
                f.write(str(score))
                f.write('\n')
    


    

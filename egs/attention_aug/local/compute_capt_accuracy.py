import argparse
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
   
# configuration.
parser.add_argument('--anno', type=str, default="capt/annotation.txt")
parser.add_argument('--pred', type=str, default="capt/prediction.txt")
parser.add_argument('--phone_table', type=str, default="data/lang/phones.txt")
parser.add_argument('--ignored_phones', type=str, default="<eps>")
parser.add_argument('--capt_dir', type=str, default="exp/models/tdnn/decode/capt")

args = parser.parse_args()

sent_anno = {}
sent_pred = {}

def process_fn(info_fn, sent_dict):
    print("process_fn:", info_fn)
    with open(info_fn, "r") as fn:
        for line1, line2, line3, line4 in itertools.zip_longest(*[fn]*4):
            ref_info = line1.split()
            utt_id = ref_info[0]
            
            ref_info = ref_info[2:]
            hyp_info = line2.split()[2:]
            op_info = line3.split()[2:]
            # ignore 4rd line (non-sense)
            assert len(ref_info) == len(hyp_info) and len(hyp_info) == len(op_info)
            
            sent_dict[utt_id] = {"ref":[], "hyp":[], "op":[]}
            # extract alignment information from the file
            for i in range(len(ref_info)):
                if op_info[i] == "I": continue
                sent_dict[utt_id]["ref"].append(ref_info[i])
                sent_dict[utt_id]["hyp"].append(hyp_info[i])
                sent_dict[utt_id]["op"].append(op_info[i])
    return sent_dict

def comp_metric(sent_anno, sent_pred):
    print("comp_metric:")
    # Detection
    anno_list = []
    pred_list = []
    # Diagnose
    diag_report = {"ce":0, "de":0, "de_ori": []}
    utt_ids = list(sent_anno.keys())
    # some details
    anno_hyp_list = []
    pred_hyp_list = []
    
    for utt_id in utt_ids:
        anno_info = sent_anno[utt_id]
        pred_info = sent_pred[utt_id]
        assert len(anno_info["op"]) == len(pred_info["op"])
        
        for i in range(len(anno_info["op"])):
            # annotation
            if anno_info["op"][i] == "C":
                anno = 1
            else:
                anno = 0
            # prediction
            if pred_info["op"][i] == "C":
                pred = 1
            else:
                pred = 0
            anno_list.append(anno)
            pred_list.append(pred)
            # diagose
            if pred == 0 and anno == 0:
                if anno_info["hyp"][i] == pred_info["hyp"][i]:                
                    diag_report["ce"] += 1
                else:
                    diag_report["de"] += 1
                    diag_report["de_ori"].append([anno_info["hyp"][i],  pred_info["hyp"][i]])
            # details
            anno_hyp_list.append(anno_info["hyp"][i])
            pred_hyp_list.append(pred_info["hyp"][i])
            
    return anno_list, pred_list, diag_report, anno_hyp_list, pred_hyp_list


def report_error_details(sent_anno, sent_pred):
    # return (MP_info):
    # { "l1_id": {"err_type": [correct_detect, correct_diagnose, total]}}
    spk2l1 = {  "ABA":"Arabic", "SKA":"Arabic", "YBAA":"Arabic", "ZHAA":"Arabic", 
                "BWC":"Mandarin", "LXC":"Mandarin", "NCC":"Mandarin", "TXHC":"Mandarin", 
                "ASI":"Hindi", "RRBI":"Hindi", "SVBI":"Hindi", "TNI":"Hindi", "HJK":"Korean", 
                "HKK":"Korean", "YDCK":"Korean", "YKWK":"Korean", "EBVS":"Spanish", 
                "ERMS":"Spanish", "MBMPS":"Spanish", "NJS":"Spanish", "HQTV":"Vietnamese", 
                "PNV":"Vietnamese", "THV":"Vietnamese","TLV":"Vietnamese"}
    utt_ids = list(sent_anno.keys())
    l1_stats = {}
    l1_errs = {}
    
    for utt_id in utt_ids:
        anno_info = sent_anno[utt_id]
        pred_info = sent_pred[utt_id]
        assert len(anno_info["op"]) == len(pred_info["op"])
        
        spk = utt_id.split("_")[0]
        l1 = spk2l1[spk]
        
        if l1 not in l1_errs:
            l1_stats[l1] = {}
            l1_errs[l1] = {}
        
        for i in range(len(anno_info["op"])):
            # MDD analysis
            err_type = anno_info["op"][i]
            
            if err_type == "S":
                err_info = "S_" + anno_info["ref"][i] + "->" + anno_info["hyp"][i] 
            elif err_type == "D":
                err_info = "D_" + anno_info["ref"][i] + "->SIL"
            else:
                continue
            
            # [correct detect, error_detect, correct diagnose, total]
            if err_info not in l1_stats[l1]:
                l1_stats[l1][err_info] = [0, 0, 0, 0]
                
            # correct detect
            if pred_info["op"][i] in ["S", "D"]:
                l1_stats[l1][err_info][0] += 1
            else:
                l1_stats[l1][err_info][1] += 1
                
            # correct diagnose
            if pred_info["hyp"][i] == anno_info["hyp"][i]:
                l1_stats[l1][err_info][2] += 1
                
            l1_stats[l1][err_info][3] += 1
                
    return l1_stats    


def plot_phn_conf_mat(anno_hyp_list, pred_hyp_list, phone_dict, capt_dir):
    num_phns = len(list(phone_dict.keys()))
    phone_list = ["stuff" for i in range(num_phns)]
    conf_mat = np.zeros((num_phns, num_phns))
    
    for phn, phn_idx in phone_dict.items():
        phone_list[phn_idx] = phn
    
    for i in range(len(anno_hyp_list)):
        anno_phn = anno_hyp_list[i]
        pred_phn = pred_hyp_list[i]
        if anno_phn in phone_dict:
            anno_idx = phone_dict[anno_phn]
        else:
            anno_idx = phone_dict["<unk>"]
        
        if pred_phn in phone_dict:
            pred_idx = phone_dict[pred_phn]
        else:
            pred_idx = phone_dict["<unk>"]
        
        conf_mat[anno_idx][pred_idx] += 1
    # normalize
    conf_mat /= conf_mat.sum(axis = 1)[:,None]
    
    df_cm = pd.DataFrame(np.round(conf_mat, 2), index = phone_list,
                         columns = phone_list)
    plt.figure(figsize = (40,40))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(capt_dir + "/phones_cm.png")

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

if __name__ == "__main__":
    sent_anno = process_fn(args.anno, sent_anno)
    sent_pred = process_fn(args.pred, sent_pred)
    anno_list, pred_list, diag_report, anno_hyp_list, pred_hyp_list = comp_metric(sent_anno, sent_pred)
    tn, fp, fn, tp = confusion_matrix(anno_list, pred_list).ravel()
    print("tn, fp, fn, tp")
    print(tn, fp, fn, tp)
    pp.pprint(classification_report(anno_list, pred_list, output_dict=True))
    print(diag_report["ce"], diag_report["de"], diag_report["ce"] / (diag_report["ce"] + diag_report["de"]))
    # print(diag_report["de_ori"])
    ignored_phones = args.ignored_phones.split(",") 
    phone_dict = {"<unk>": 0}
    
    # phones' confusion matrix
    with open(args.phone_table, "r") as phn_fn:
        for line in phn_fn.readlines():
            info = line.split()
            if info[0] in ignored_phones:
                continue
            
            phn = info[0]            
            if phn not in phone_dict:
                phone_dict[phn] = len(list(phone_dict.keys()))
    
    # plot_phn_conf_mat(anno_hyp_list, pred_hyp_list, phone_dict, args.capt_dir)
    l1_stats = report_error_details(sent_anno, sent_pred)
    err_types = {}
    
    # speaker-wise information
    with open(args.capt_dir + "/per_l1.csv", "w") as fn:
        fn.write("l1, err_type, cdetect, edetect, cdiagnose, total\n")
        for l1_id in list(l1_stats.keys()):
            l1_err_types = l1_stats[l1_id]
            for err_type in list(l1_err_types.keys()):
                info = l1_id + "," + err_type
                err_stats = l1_err_types[err_type]
                
                if err_type not in err_types:
                    err_types[err_type] = [0, 0, 0, 0]
                
                for i in range(len(err_types[err_type])):
                    err_types[err_type][i] += err_stats[i]
                    info += "," + str(err_stats[i])
                fn.write(info + "\n")
    
    # corpus-wise information
    with open(args.capt_dir + "/per_all.csv", "w") as fn:
        fn.write("err_type, cdetect, edetect, cdiagnose, total\n")
        for err_type in list(err_types.keys()):
            info = err_type
            err_stats = err_types[err_type]
                
            for i in range(len(err_types[err_type])):
                info += "," + str(err_stats[i])
            fn.write(info + "\n")

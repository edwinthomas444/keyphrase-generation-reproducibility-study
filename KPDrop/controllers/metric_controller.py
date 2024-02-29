from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.scheme import IOB2
# from utils.conlleval import evaluate


def compute_F1(prec, rec):
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

def metric_fn(items, config):
    # each item is for a single mini-batch , together all items constitute the whole dataset
    metrics = [item["metrics"] for item in items]
    if config["display_metric"] == "accuracy":
        correct_predictions = sum([metric["correct_predictions"] for metric in metrics])
        total = sum([metric["total"] for metric in metrics])
        accuracy = correct_predictions/total if total > 0 else 0
        loss = sum([metric["loss"] for metric in metrics])/len(metrics) if len(metrics) > 0 else 0

        composed_metric = {"loss": loss,
                           "accuracy": accuracy*100}

    elif config["display_metric"] == "F1" and config["model_type"] == "seq_label":
        loss = sum([metric["loss"] for metric in metrics]) / len(metrics) if len(metrics) > 0 else 0
        display_items = [item["display_items"] for item in items]
        all_predictions = []
        all_labels = []
        for item in display_items:
            all_predictions += item["predictions"]
            all_labels += item["labels"]

        composed_metric = {"loss": loss,
                           "F1": seqeval_f1_score(all_labels, all_predictions, scheme=IOB2)}
    elif config["model_type"] == "seq2seq" or config["model_type"] == "seq2set":
        composed_metric = {}
        total_data = sum([metric["total_data"] for metric in metrics])
        for beam in [False]:
            if beam:
                beam_tag = "_beam"
            else:
                beam_tag = ""

            if "total_present_precision_beam" not in metrics[0] and beam:
                continue

            if isinstance(metrics[0]["total_present_precision"], int):
                continue
            
            eps = 1e-08
            for topk in metrics[0]["total_present_precision"]:
                total_present_precision = sum(
                    [metric["total_present_precision" + beam_tag][topk] for metric in metrics])
                
                
                total_present_recall = sum([metric["total_present_recall" + beam_tag][topk] for metric in metrics])
                avg_present_precision = total_present_precision / total_data
                avg_present_recall = total_present_recall / total_data
                macro_present_F1 = compute_F1(avg_present_precision, avg_present_recall)

                composed_metric["macro_present_precision_" + topk + beam_tag] = avg_present_precision
                composed_metric["macro_present_recall_" + topk + beam_tag] = avg_present_recall
                composed_metric["macro_present_F1_" + topk + beam_tag] = macro_present_F1

                # merge all batch lists to one list
                composed_metric["macro_f_present_list_"+ topk + beam_tag] = []
                composed_metric["macro_f_absent_list_"+ topk + beam_tag] = []
                # print('\n len metrics: ', len(metrics))


                # total_length = 0
                for metric in metrics:
                    # a list of size minibatch
                    # total_length+= len(metric["macro_f_present_list" + beam_tag][topk])
                    composed_metric["macro_f_present_list_"+ topk + beam_tag].extend(metric["macro_f_present_list" + beam_tag][topk])
                    composed_metric["macro_f_absent_list_"+ topk + beam_tag].extend(metric["macro_f_absent_list" + beam_tag][topk])
                
                # print('\n total length: ', total_length)


                # micro scores computation
                total_present_tp = sum([metric["total_present_tp" + beam_tag][topk] for metric in metrics])
                total_present_prediction = sum([metric["total_present_prediction" + beam_tag][topk] for metric in metrics]) + eps
                total_present_trg = sum([metric["total_present_trg" + beam_tag][topk] for metric in metrics]) + eps
                micro_present_precision = total_present_tp/total_present_prediction
                micro_present_recall = total_present_tp/total_present_trg
                micro_present_F1 = compute_F1(micro_present_precision, micro_present_recall)

                composed_metric["micro_present_precision_" + topk + beam_tag] = micro_present_precision
                composed_metric["micro_present_recall_" + topk + beam_tag] = micro_present_recall
                composed_metric["micro_present_F1_" + topk + beam_tag] = micro_present_F1



                total_absent_precision = sum([metric["total_absent_precision" + beam_tag][topk] for metric in metrics])
                total_absent_recall = sum([metric["total_absent_recall" + beam_tag][topk] for metric in metrics])
                avg_absent_precision = total_absent_precision / total_data
                avg_absent_recall = total_absent_recall / total_data
                macro_absent_F1 = compute_F1(avg_absent_precision, avg_absent_recall)

                composed_metric["macro_absent_precision_" + topk + beam_tag] = avg_absent_precision
                composed_metric["macro_absent_recall_" + topk + beam_tag] = avg_absent_recall
                composed_metric["macro_absent_F1_" + topk + beam_tag] = macro_absent_F1

                # micro scores computation
                total_absent_tp = sum([metric["total_absent_tp" + beam_tag][topk] for metric in metrics])
                total_absent_prediction = sum([metric["total_absent_prediction" + beam_tag][topk] for metric in metrics]) + eps
                total_absent_trg = sum([metric["total_absent_trg" + beam_tag][topk] for metric in metrics]) + eps
                micro_absent_precision = total_absent_tp/total_absent_prediction
                micro_absent_recall = total_absent_tp/total_absent_trg
                micro_absent_F1 = compute_F1(micro_absent_precision, micro_absent_recall)

                composed_metric["micro_absent_precision_" + topk + beam_tag] = micro_absent_precision
                composed_metric["micro_absent_recall_" + topk + beam_tag] = micro_absent_recall
                composed_metric["micro_absent_F1_" + topk + beam_tag] = micro_absent_F1

        loss = sum([metric["loss"] * metric["total_data"] for metric in metrics]) / total_data
        composed_metric["loss"] = loss

    return composed_metric


def compose_dev_metric(metrics, config):
    total_metric = 0
    n = len(metrics)
    for key in metrics:
        total_metric += metrics[key][config["save_by"]]
    return config["metric_direction"] * total_metric / n

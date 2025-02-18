from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_multiclass(y_test, y_pred, class_names):
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # 计算每个类别的正确率
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        correct = cm[i, i]  # 正确分类的样本数
        total = np.sum(cm[i, :])  # 该类别的总样本数
        class_accuracy[class_name] = correct / total  # 正确率
        print(f"Accuracy for {class_name}: {class_accuracy[class_name]:.4f}")
    class_names = list(class_accuracy.keys())
    accuracies = list(class_accuracy.values())

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, accuracies, color='skyblue',width=0.5)

    # 添加标题和标签
    plt.title('Classwise Accuracy', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    # 设置旋转角度，以便更好地显示类别名称
    plt.xticks(rotation=25, ha='right')

    # 显示图形
    plt.tight_layout()  # 调整布局，防止标签被截断
    plt.savefig('classwise_accuracy.png')
    plt.show()
    # 计算全局准确率
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # 计算精确率、召回率和F1分数
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 打印计算结果
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 可视化混淆矩阵
    # try:
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')
    #     plt.title('Confusion Matrix')
    #     plt.show()
    # except ImportError:
    #     print("Seaborn or matplotlib not available for heatmap visualization.")
    
    # 计算 ROC 曲线和 AUC（仅适用于二分类）
    # if len(class_names) == 2:
    #     fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_pred)
    #     roc_auc = auc(fpr_roc, tpr_roc)
        
    #     # 绘制 ROC 曲线
    #     plt.figure(figsize=(6, 5))
    #     plt.plot(fpr_roc, tpr_roc, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    #     plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic (ROC) Curve')
    #     plt.legend(loc='lower right')
    #     plt.show()
    # else:
    #     print("ROC curve is not applicable for multi-class classification.")

# 示例使用
# y_test = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
# class_names = ['Class 0', 'Class 1', 'Class 2']
# evaluate_multiclass(y_test, y_pred, class_names)
    # 从文件加载预测结果
def load_predictions(file_path):
    # 从文件加载预测结果
    df = pd.read_csv(file_path)
    y_test_bin = df['True Labels'].values
    y_pred = df['Predicted Labels'].values
    return y_test_bin, y_pred
def evaluate(y_test_bin, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_test_bin, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # 计算假阳性率 (FPR) 和假阴性率 (FNR)
    tn, fp, fn, tp = cm.ravel()  # 将混淆矩阵分解为四个部分
    
    # 假阳性率 (FPR)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    prc=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*prc*recall/(prc+recall)


    fpr = fp / (fp + tn)
    
    # 假阴性率 (FNR)
    fnr = fn / (fn + tp)
    
    # 误报率 (FAR)
    far = fp / (fp + tp)

    # 打印计算结果
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {prc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"False Alarm Rate (FAR): {far:.4f}")

y_test_bin, y_pred = load_predictions('xgb-20predictions.csv')
# y_test_bin, y_pred = load_predictions('42predictions.csv')
# y_test_bin, y_pred = load_predictions('m-42predictions.csv')
# y_test_bin, y_pred = load_predictions('m-20predictions.csv')
class_names =['Analysis', 'Backdoor', 'DoS' ,'Exploits', 'Fuzzers' ,'Generic', 'Normal','Reconnaissance' ,'Shellcode', 'Worms']
# evaluate_multiclass(y_test_bin, y_pred, class_names)
evaluate(y_test_bin, y_pred)
# evaluate(y_test_bin, y_pred)

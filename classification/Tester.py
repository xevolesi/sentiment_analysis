class Tester:
    @staticmethod
    def test(clsr_obj, test_df):
        wrong_answers = 0
        for row in test_df.iterrows():
            predicted_label = clsr_obj.classify_text(
                row[1][clsr_obj.model.tcn])
            if predicted_label != row[1][clsr_obj.model.lcn]:
                wrong_answers += 1
        return wrong_answers / len(test_df)

    @staticmethod
    def get_summary(clsr_obj, test_df):
        wrong_answers = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for row in test_df.iterrows():
            predicted_label = clsr_obj.classify_text(row[1][clsr_obj.model.tcn])

            if predicted_label != row[1][clsr_obj.model.lcn]:
                wrong_answers += 1
                if predicted_label == "pos":
                    FP += 1
                if predicted_label == "neg":
                    FN += 1
            else:
                if predicted_label == "pos":
                    TP += 1
                if predicted_label == "neg":
                    TN += 1
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F_1 = (2 * Precision * Recall) / (Precision + Recall)
        return {
                "Error:": wrong_answers / len(test_df),
                "True positive": TP,
                "True negative": TN,
                "False positive": FP,
                "False negative": FN,
                "Precision": Precision,
                "Recall": Recall,
                "F-1 measure": F_1
            }

import pandas as pd


def results_to_pred_ground_truth_csv(file_pred_participant, output_path):

    df_pred_participant = pd.read_csv(file_pred_participant)
    df_corr_prot_id = pd.read_csv("corr_prot_id_testset.csv")
    df_ground_truth = pd.read_csv("test_set.csv")
    list_pred_ground_truth = []

    for i in range(df_pred_participant.shape[0]):
        test_prot_id = df_pred_participant.at[i, "protein_id"]
        line_init_prot_id = df_corr_prot_id.loc[
            df_corr_prot_id["ANONYMID"] == test_prot_id
        ]
        init_prot_id = line_init_prot_id["PDBID"].item()
        pred_class = df_pred_participant.at[i, "class_id"]
        line_ground_truth_class = df_ground_truth.loc[
            df_ground_truth["protein_id"] == init_prot_id
        ]
        #print(init_prot_id)
        ground_truth_class = line_ground_truth_class["class_id"].item()
        list_pred_ground_truth.append(
            [test_prot_id, init_prot_id, pred_class, ground_truth_class]
        )
        df_pred_ground_truth = pd.DataFrame(
            list_pred_ground_truth,
            columns=[
                "test_prot_id",
                "init_prot_id",
                "pred_class",
                "ground_truth_class",
            ],
        )

    df_pred_ground_truth.to_csv(output_path, header=True, index=False)

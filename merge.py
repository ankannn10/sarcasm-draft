import argparse
import pandas as pd

def merge_predictions(emotion_csv, sarcasm_csv, output_csv):
    """
    Merge emotion and sarcasm (irony) predictions for visualization.

    The emotion CSV is expected to have:
        - comment
        - comment_emotion1, comment_emotion2, comment_emotion3,
          comment_conf1, comment_conf2, comment_conf3, etc.
    
    The sarcasm CSV is expected to have:
        - Comment, Most Relevant Transcript Chunk, sarcasm_label, sarcasm_prob

    This function will:
      - Standardize the key column names.
      - Merge the two dataframes on the comment text.
      - Rename 'sarcasm_label' to 'Irony' and 'sarcasm_prob' to 'Irony Probability'.
      - Output the merged dataframe to a new CSV.
    """
    # Load the emotion predictions CSV
    emotion_df = pd.read_csv(emotion_csv)
    # Load the sarcasm (irony) predictions CSV
    sarcasm_df = pd.read_csv(sarcasm_csv)
    
    # Standardize key column names:
    # In emotion_df, the key is 'comment'
    # In sarcasm_df, the key is 'Comment'
    if "comment" not in emotion_df.columns and "Comment" in emotion_df.columns:
        emotion_df.rename(columns={"Comment": "comment"}, inplace=True)
    
    if "Comment" not in sarcasm_df.columns and "comment" in sarcasm_df.columns:
        sarcasm_df.rename(columns={"comment": "Comment"}, inplace=True)
    
    # Merge the two DataFrames on the comment text.
    merged_df = pd.merge(emotion_df, sarcasm_df, left_on="comment", right_on="Comment", how="inner")
    
    # Optionally, drop the duplicate comment column from sarcasm_df
    merged_df.drop(columns=["Comment"], inplace=True)
    
    # Rename sarcasm columns for visualization purposes
    merged_df.rename(columns={
        "sarcasm_label": "Irony",
        "sarcasm_prob": "Irony Probability"
    }, inplace=True)
    
    # Save the merged DataFrame
    merged_df.to_csv(output_csv, index=False)
    print(f"Final merged predictions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge emotion and sarcasm (irony) predictions for visualization."
    )
    parser.add_argument("--emotion_csv", type=str, required=True,
                        help="Path to the emotion predictions CSV file")
    parser.add_argument("--sarcasm_csv", type=str, required=True,
                        help="Path to the sarcasm (irony) predictions CSV file")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to output the merged CSV file")
    
    args = parser.parse_args()
    merge_predictions(args.emotion_csv, args.sarcasm_csv, args.output_csv)

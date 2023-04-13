output_dir = "../saved_models/"
figure_folder = "../figures/"

# dataset paths
train_df_path =         "train/labels_1_to_8_comma.csv"
train_image_base_path = "train/combined_1_to_8_comma_real/"
test_df_path =          "test/labels_1_to_8_comma.csv"
test_image_base_path =  "test/combined_1_to_8_comma_real/"
test_image_actual_shape_base_path = "test/combined_1_to_8_comma_actual_shape_real/"



model_folder_name =  "multi_digit_model_1_to_8_comma_transformer"
model_json_file_name =  "multi_digit_model_1_to_8_comma_transformer_json.json"
model_weights_file_name =  "multi_digit_model_1_to_8_comma_transformer_weights.h5"
model_tflite_name =  "worksheet.multi_digit_model_1_to_8_comma_transformer.tflite"

# dataloader params
img_height = 28
img_width = 168
num_time_steps = 42  # img_width//4
max_digit_length = 11
shuffle = True

# model params
num_classes = 12
# batch_size = 256
batch_size = 128
epochs = 10
early_stopping_patience = 3



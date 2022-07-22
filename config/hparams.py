import random
from collections import defaultdict

HPARAMS = defaultdict(
    # Dataset reader arguments
    img_feature_type = "dan_faster_rcnn_x101",                        # faster_rcnn_x101, dan_faster_rcnn_x101
    img_features_h5 = "data/visdial_1.0_img/features_%s_%s.h5",       # img_feature_type | train, val, test
    imgid2idx_path = "data/visdial_1.0_img/%s_imgid2idx.pkl",         # dan_img - train, val, test

    questions_json = "data/v2/v2_OpenEnded_mscoco_%s_questions.json", # train2014, val2014, test2015, test-dev2015
    annotations_json = "data/v2/v2_mscoco_%s_annotations.json",       # train2014, val2014
    question_types_txt = "data/v2/mscoco_question_types.txt",
    vqa_word_counts_json = "data/v2/v2_word_counts_train.json",
    answer_to_index_json = "data/v2/v2_answer_to_index_train.json",
    index_to_answer_json = "data/v2/v2_index_to_answer_train.json",
    vqa_text_features_hdf5 = "data/vqa_v2_text/vqa_v2_text_%s.hdf5",  # train2014, val2014, test2015

    visdial_json = "data/v1.0/visdial_1.0_%s.json",
    valid_dense_json = "data/v1.0/visdial_1.0_val_dense_annotations.json",
    visdial_word_counts_json = "data/v1.0/visdial_1.0_word_counts_train.json", 
    visdial_text_features_hdf5 = "data/visdial_1.0_text/visdial_1.0_text_%s.hdf5",
    
    share_word_counts_json = "data/share_text/share_word_counts_train.json",
    pretrained_glove = 'data/share_text/glove.840B.300d.txt',
    glove_npy = 'data/share_text/glove.npy',

    # Model save arguments
    root_dir = "", # for saving logs, checkpoints and result files
    save_dirpath = "checkpoints",
    load_pthpath = "",
    result_dirpath = "results",
    fig_dirpath = "fig",

    # Preprocess related arguments
    num_answers = 1000,
    num_ans_types = 3,
    num_ques_types = 65,
    max_question_length = 12,
    min_answer_length = 3,
    max_answer_length = 8,
    max_round_history = 3,
    vocab_min_count = 5,
    overfit = False,
    num_pick_data = 20000,

    # Train related arguments
    num_samples = 25,
    accumulation_steps = 32,
    gpu_ids = [0],
    cpu_workers = 8,
    tensorboard_step = 100,
    random_seed=random.sample(range(1000, 10000), 1),

    # Opitimization related arguments
    num_epochs = 48,
    train_batch_size = 1, # vqa: 100 # 32 x num_gpus is a good rule of thumb
    eval_batch_size = 100,
    training_splits = "train",
    evaluation_type = "disc_gen",
    lr_scheduler = True,
    lr_scheduler_step = 200,
    warmup_epochs = 1,
    warmup_factor = 0.1,
    initial_lr = 1e-5,
    lr_gamma = 0.1,
    lr_milestones = [5, 10],  # epochs when lr —> lr * lr_gamma
    initial_beta = 0.1,
    gamma = 0.5,
    num_optim_mi = 2,

    # Model related arguments
    encoder = "mvan",
    decoder = "disc_gen",  # [disc,gen]
    aggregation_type = "average",
    word_embedding_size = 300,

    add_positional_encoding = False,
    spatial_feat = True,
    img_feat_size = 2048,
    img_sp_feat_size = 6,
    img_hidden_size = 1024,
    sent_emb_size = 768,
    hist_emb_size = 512,
    cont_emb_size = 512,
    type_emb_size = 512,
    fusion_out_size = 512,
    lstm_hidden_size = 512,
    lstm_num_layers = 1,
    dropout = 0.4,
    dropout_fc = 0.1,
    
    num_heads = 1,
    top_k = 32,
)
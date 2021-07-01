import argparse
import numpy as np
from main import main


# -data blog -is_regression False -clean_text True -tokenizer simple -use_pretrained_emb False -save_test_pred False
# -is_adv True -model linear_adv -learnable_embeddings False
# -embeddings ../../../storage/glove_300_dim/simple_glove_three.vec -epochs 130
# -is_post_hoc False -use_wandb False -experiment_name delete -adv_loss_scale 2.5 -mode_of_loss_scale exp
# -training_loop_type three_phase_custom -trim_data True -eps 15.0 -eps_scale constant -optimizer sgd -lr 0.02
# -fair_grad False -use_adv_dataset True -bs 64 -fairness_function demographic_parity -fairness_score_function grms -noise_layer False -sample_specific_class True -only_perturbate True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some data')
    parser.add_argument('first', metavar='N', type=float,
                        help='an integer for the accumulator')
    parser.add_argument('second', metavar='N', type=float,
                        help='an integer for the accumulator')
    args = parser.parse_args()
    print(args.first)
    print(args.second)

    adv_scale = 1.0
    noise_layer = False
    eps = 10.0
    bs = 64
    only_perturbate = True
    mode_of_loss_scale = 'exp'
    optimizer = 'sgd'
    lrs = [0.02, 0.01]
    epochs = 15
    is_adv = True

    assert args.second > args.first

 # epss = [8.0,20.0]
# epss = [25.0, 50.0, 100.0, 500.0, 1000.0]
epss = [1.0]
# adv_scales = [round(i,2) for i in np.arange(0.1,0.5,0.1)]
# adv_scales = [round(i,2) for i in np.arange(0.5,1.0,0.1)]
adv_scales = [round(i,2) for i in np.arange(args.first,args.second,0.1)]
# adv_scales = [round(i,2) for i in np.arange(1.5,2.0,0.1)]
# adv_scales = [round(i,2) for i in np.arange(2.0,2.6,0.1)]
# adv_scales = [1.0]
# mode_of_loss_scales = ['exp', 'constant']
mode_of_loss_scales = ['exp']

for eps in epss:
     for lr in lrs:
         for adv_scale in adv_scales:
             for mode_of_loss_scale in mode_of_loss_scales:
                 print("******", adv_scale, mode_of_loss_scale, lr, eps)
                 try:
                     main(emb_dim=300,
                              spacy_model="en_core_web_sm",
                              seed=1234,
                              dataset_name='encoded_emoji',
                              batch_size=bs,
                              pad_token='<pad>',
                              unk_token='<unk>',
                              pre_trained_embeddings='../../bias-in-nlp/different_embeddings/simple_glove_vectors.vec',
                              model_save_name='bilstm.pt',
                              model='linear_adv',
                              regression=False,
                              tokenizer_type='simple',
                              use_clean_text=False,
                              max_length=None,
                              epochs=epochs,
                              learnable_embeddings=False,
                              vocab_location=False,
                              is_adv=is_adv,
                              adv_loss_scale=adv_scale,
                              use_pretrained_emb=False ,
                              default_emb_dim=300,
                              save_test_pred=False,
                              noise_layer=noise_layer,
                              eps=eps,
                              is_post_hoc=False,
                              train_main_model=True,
                              use_wandb=False,
                              config_dict="simple",
                              experiment_name="hyper-param-search",
                              only_perturbate=only_perturbate,
                              mode_of_loss_scale=mode_of_loss_scale,
                              training_loop_type='three_phase_custom',
                              hidden_loss=False,
                              hidden_l1_scale=0.5,
                              hidden_l2_scale=0.5,
                              reset_classifier=False,
                              reset_adv=True,
                              encoder_learning_rate_second_phase=0.01,
                              classifier_learning_rate_second_phase=0.01,
                              trim_data=False,
                              eps_scale='constant',
                              optimizer=optimizer,
                              lr=lr,
                              fair_grad=False,
                              reset_fairness=False,
                              use_adv_dataset=True,
                              use_lr_schedule=True,
                              fairness_function='demographic_parity',
                              fairness_score_function='grms',
                              sample_specific_class=False
                              )
                 except:
                     continue
if [ -z $TTS_DATA_ROOT ]; then
  echo "環境変数「TTS_DATA_ROOT」が設定されていません"
  exit -1
fi

if [ -z $1 ]; then
  echo "第一引数にrecipe名を指定してください"
  exit -1
fi

if [ ! -e ${TTS_DATA_ROOT}/$1 ];
then
    echo "指定された名前のレシピデータディレクトリが見つかりません：$1"
    exit -1
fi
. $ESPNET_ROOT/egs2/TEMPLATE/tts1/setup.sh $ESPNET_ROOT/egs2/$1/tts1

cat <<'EOF' >> $ESPNET_ROOT/egs2/$1/tts1/local/data.sh
#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=2
threshold=45
nj=6
fs=24000

log "$0 $*"
# shellcheck disable=SC1091
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

# shellcheck disable=SC1091
. ./path.sh || exit 1;
# shellcheck disable=SC1091
. ./cmd.sh || exit 1;
# shellcheck disable=SC1091
. ./db.sh || exit 1;

character=`pwd`
character=`dirname $character`
character=${character##*/}

if [ -z "$TTS_DATA_ROOT/$character" ]; then
   log "'$character' is not found."
   exit 1
fi

db_root=$TTS_DATA_ROOT
train_set=tr_no_dev
dev_set=dev
eval_set=eval1



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh "${db_root}/${character}" data/all ${fs}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: scripts/audio/trim_silence.sh"
    # shellcheck disable=SC2154
    scripts/audio/trim_silence.sh \
        --cmd "${train_cmd}" \
        --nj "${nj}" \
        --fs ${fs} \
        --win_length 2048 \
        --shift_length 512 \
        --threshold "${threshold}" \
        data/all data/all/log
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    log "include eval dev data for small dataset"
    utils/subset_data_dir.sh data/all 24 data/deveval
    utils/subset_data_dir.sh --first data/deveval 12 "data/${dev_set}"
    utils/subset_data_dir.sh --last data/deveval 12 "data/${eval_set}"
    utils/copy_data_dir.sh data/all "data/${train_set}"
    utils/fix_data_dir.sh "data/${train_set}"
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
EOF

cat <<'EOF' >> $ESPNET_ROOT/egs2/$1/tts1/local/data_prep.sh
#!/usr/bin/env bash

db_root=$1
data_dir=$2
fs=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <data_dir> <fs>"
    echo "e.g.: $0 downloads/jsut data/train 44100"
    exit 1
fi

set -euo pipefail

# check directory existence
[ ! -e "${data_dir}" ] && mkdir -p "${data_dir}"

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${utt2spk}" ] && rm "${utt2spk}"
[ -e "${text}" ] && rm "${text}"

# make scp, utt2spk, and spk2utt
ds_list=$(ls ${db_root})
for ds in $ds_list
do
    spk_list=$(ls ${db_root}/${ds})
    for spk in $spk_list
    do
        style_list=$(ls ${db_root}/${ds}/${spk})
        for style in $style_list
        do
            echo "${db_root}/${ds}/${spk}/${style}"
            find "${db_root}/${ds}/${spk}/${style}/wav" -name "*.wav" | sort | while read -r filename; do
                id=${ds}_${spk}_${style}_$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
                echo "${id} sox \"${filename}\" -r ${fs} -t wav -c 1 -b 16 - |" >> "${scp}"
                echo "${id} ${ds}_${spk}" >> "${utt2spk}"
            done
        done
    done
done
utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"
echo "Successfully finished making wav.scp, utt2spk, spk2utt."

# make text
for ds in $ds_list
do
    spk_list=$(ls ${db_root}/${ds})
    for spk in $spk_list
    do
        style_list=$(ls ${db_root}/${ds}/${spk})
        for style in $style_list
        do
            echo "${db_root}/${ds}/${spk}/${style}"
            find "${db_root}/${ds}/${spk}/${style}" -name "*.txt" | grep "transcripts_utf8.txt" | while read -r filename; do
                awk -F ":" -v header=${ds}_${spk}_${style} '{print header "_" $1 " " $2}' < "${filename}" | sort >> "${text}"
            done
        done
    done
done
echo "Successfully finished making text."

utils/fix_data_dir.sh "${data_dir}"
echo "Successfully finished preparing data directory."

EOF

chmod +x $ESPNET_ROOT/egs2/$1/tts1/local/data.sh
chmod +x $ESPNET_ROOT/egs2/$1/tts1/local/data_prep.sh



cp -r  $ESPNET_ROOT/egs2/tsukuyomi/tts1/conf/tuning $ESPNET_ROOT/egs2/$1/tts1/conf
cp $ESPNET_ROOT/egs2/tsukuyomi/tts1/run.sh $ESPNET_ROOT/egs2/$1/tts1/

# decrease batch_bins to 100,000 from 10,000,000
cat <<'EOF' >> $ESPNET_ROOT/egs2/$1/tts1/conf/tuning/train_full_band_jets_gst_spk_harvest.yaml
# This configuration is for ESPnet2 to train JETS, which
# is truely end-to-end text-to-waveform model. To run
# this config, you need to specify "--tts_task gan_tts"
# option for tts.sh at least and use 44100 hz audio as
# the training data (mainly tested on No.7 Rohan normal).
# This configuration tested on a GPU (RTX 3060) with 12GB GPU
# memory. It takes around 1.5 weeks to finish the training
# but 300k iters model should generate reasonable results.

##########################################################
#                  TTS MODEL SETTING                     #
##########################################################
tts: jets
tts_conf:
    # generator related
    generator_type: jets_generator
    generator_params:
        adim: 384                                    # attention dimension
        aheads: 2                                    # number of attention heads
        elayers: 4                                   # number of encoder layers
        eunits: 1536                                 # number of encoder ff units
        dlayers: 4                                   # number of decoder layers
        dunits: 1536                                 # number of decoder ff units
        positionwise_layer_type: conv1d              # type of position-wise layer
        positionwise_conv_kernel_size: 3             # kernel size of position wise conv layer
        duration_predictor_layers: 2                 # number of layers of duration predictor
        duration_predictor_chans: 256                # number of channels of duration predictor
        duration_predictor_kernel_size: 3            # filter size of duration predictor
        use_masking: True                            # whether to apply masking for padded part in loss calculation
        encoder_normalize_before: True               # whether to perform layer normalization before the input
        decoder_normalize_before: True               # whether to perform layer normalization before the input
        encoder_type: transformer                    # encoder type
        decoder_type: conformer                    # decoder type
        conformer_rel_pos_type: latest               # relative positional encoding type
        conformer_pos_enc_layer_type: rel_pos        # conformer positional encoding type
        conformer_self_attn_layer_type: rel_selfattn # conformer self-attention type
        conformer_activation_type: swish             # conformer activation type
        use_macaron_style_in_conformer: true         # whether to use macaron style in conformer
        use_cnn_in_conformer: true                   # whether to use CNN in conformer
        conformer_enc_kernel_size: 7                 # kernel size in CNN module of conformer-based encoder
        conformer_dec_kernel_size: 31                # kernel size in CNN module of conformer-based decoder
        init_type: xavier_uniform                    # initialization type
        transformer_enc_dropout_rate: 0.2            # dropout rate for transformer encoder layer
        transformer_enc_positional_dropout_rate: 0.2 # dropout rate for transformer encoder positional encoding
        transformer_enc_attn_dropout_rate: 0.2       # dropout rate for transformer encoder attention layer
        transformer_dec_dropout_rate: 0.2            # dropout rate for transformer decoder layer
        transformer_dec_positional_dropout_rate: 0.2 # dropout rate for transformer decoder positional encoding
        transformer_dec_attn_dropout_rate: 0.2       # dropout rate for transformer decoder attention layer
        pitch_predictor_layers: 5                    # number of conv layers in pitch predictor
        pitch_predictor_chans: 256                   # number of channels of conv layers in pitch predictor
        pitch_predictor_kernel_size: 5               # kernel size of conv leyers in pitch predictor
        pitch_predictor_dropout: 0.5                 # dropout rate in pitch predictor
        pitch_embed_kernel_size: 1                   # kernel size of conv embedding layer for pitch
        pitch_embed_dropout: 0.0                     # dropout rate after conv embedding layer for pitch
        stop_gradient_from_pitch_predictor: true     # whether to stop the gradient from pitch predictor to encoder
        energy_predictor_layers: 2                   # number of conv layers in energy predictor
        energy_predictor_chans: 256                  # number of channels of conv layers in energy predictor
        energy_predictor_kernel_size: 3              # kernel size of conv leyers in energy predictor
        energy_predictor_dropout: 0.5                # dropout rate in energy predictor
        energy_embed_kernel_size: 1                  # kernel size of conv embedding layer for energy
        energy_embed_dropout: 0.0                    # dropout rate after conv embedding layer for energy
        stop_gradient_from_energy_predictor: false   # whether to stop the gradient from energy predictor to encoder
        spks: 128 
        generator_out_channels: 1
        generator_channels: 512
        generator_global_channels: -1
        generator_kernel_size: 7
        generator_upsample_scales: [8, 8, 2, 2, 2]
        generator_upsample_kernel_sizes: [16, 16, 4, 4, 4]
        generator_resblock_kernel_sizes: [3, 7, 11]
        generator_resblock_dilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        generator_use_additional_convs: true
        generator_bias: true
        generator_nonlinear_activation: "LeakyReLU"
        generator_nonlinear_activation_params:
            negative_slope: 0.1
        generator_use_weight_norm: true
        segment_size: 32                             # segment size for random windowed discriminator

    # discriminator related
    discriminator_type: hifigan_multi_scale_multi_period_discriminator
    discriminator_params:
        scales: 1
        scale_downsample_pooling: "AvgPool1d"
        scale_downsample_pooling_params:
            kernel_size: 4
            stride: 2
            padding: 2
        scale_discriminator_params:
            in_channels: 1
            out_channels: 1
            kernel_sizes: [15, 41, 5, 3]
            channels: 128
            max_downsample_channels: 1024
            max_groups: 16
            bias: True
            downsample_scales: [2, 2, 4, 4, 1]
            nonlinear_activation: "LeakyReLU"
            nonlinear_activation_params:
                negative_slope: 0.1
            use_weight_norm: True
            use_spectral_norm: False
        follow_official_norm: False
        periods: [2, 3, 5, 7, 11]
        period_discriminator_params:
            in_channels: 1
            out_channels: 1
            kernel_sizes: [5, 3]
            channels: 32
            downsample_scales: [3, 3, 3, 3, 1]
            max_downsample_channels: 1024
            bias: True
            nonlinear_activation: "LeakyReLU"
            nonlinear_activation_params:
                negative_slope: 0.1
            use_weight_norm: True
            use_spectral_norm: False

    # loss function related
    generator_adv_loss_params:
        average_by_discriminators: false # whether to average loss value by #discriminators
        loss_type: mse                   # loss type, "mse" or "hinge"
    discriminator_adv_loss_params:
        average_by_discriminators: false # whether to average loss value by #discriminators
        loss_type: mse                   # loss type, "mse" or "hinge"
    feat_match_loss_params:
        average_by_discriminators: false # whether to average loss value by #discriminators
        average_by_layers: false         # whether to average loss value by #layers of each discriminator
        include_final_outputs: true      # whether to include final outputs for loss calculation
    mel_loss_params:
        fs: 44100                        # must be the same as the training data
        n_fft: 2048                      # fft points
        hop_length: 512                  # hop size
        win_length: null                 # window length
        window: hann                     # window type
        n_mels: 80                       # number of Mel basis
        fmin: 0                          # minimum frequency for Mel basis
        fmax: null                       # maximum frequency for Mel basis
        log_base: null                   # null represent natural log
    lambda_adv: 1.0                      # loss scaling coefficient for adversarial loss
    lambda_mel: 45.0                     # loss scaling coefficient for Mel loss
    lambda_feat_match: 2.0               # loss scaling coefficient for feat match loss
    lambda_var: 1.0
    lambda_align: 2.0
    # others
    sampling_rate: 44100                 # needed in the inference for saving wav
    cache_generator_outputs: true        # whether to cache generator outputs in the training
    # reduction_factor: 1

# extra module for additional inputs
pitch_extract: dio           # pitch extractor type
pitch_extract_conf:
    reduction_factor: 1
    use_token_averaged_f0: false
    f0min: 80
    f0max: 1200
pitch_normalize: global_mvn  # normalizer for the pitch feature
energy_extract: energy       # energy extractor type
energy_extract_conf:
    reduction_factor: 1
    use_token_averaged_energy: False
energy_normalize: global_mvn # normalizer for the energy feature

##########################################################
#            OPTIMIZER & SCHEDULER SETTING               #
##########################################################
# optimizer setting for generator
optim: adamw
optim_conf:
    lr: 2.0e-4
    betas: [0.8, 0.99]
    eps: 1.0e-9
    weight_decay: 0.0
scheduler: exponentiallr
scheduler_conf:
    gamma: 0.999875
# optimizer setting for discriminator
optim2: adamw
optim2_conf:
    lr: 2.0e-4
    betas: [0.8, 0.99]
    eps: 1.0e-9
    weight_decay: 0.0
scheduler2: exponentiallr
scheduler2_conf:
    gamma: 0.999875
generator_first: true # whether to start updating generator first

##########################################################
#                OTHER TRAINING SETTING                  #
##########################################################
num_iters_per_epoch: 1000 # number of iterations per epoch
max_epoch: 1000           # number of epochs
accum_grad: 1             # gradient accumulation
batch_bins: 320000       # batch bins (feats_type=raw)
batch_type: numel         # how to make batch
grad_clip: -1             # gradient clipping norm
grad_noise: false         # whether to use gradient noise injection
sort_in_batch: descending # how to sort data in making batch
sort_batch: descending    # how to sort created batches
num_workers: 2            # number of workers of data loader
use_amp: false            # whether to use pytorch amp
log_interval: 50          # log interval in iterations
keep_nbest_models: 5      # number of models to keep
num_att_plot: 5           # number of attention figures to be saved in every check
seed: 777                 # random seed number
patience: null            # patience for early stopping
unused_parameters: true   # needed for multi gpu case
best_model_criterion:     # criterion to save the best models
-   - valid
    - text2mel_loss
    - min
-   - train
    - text2mel_loss
    - min
-   - train
    - total_count
    - max
cudnn_deterministic: false # setting to false accelerates the training speed but makes it non-deterministic
                           # in the case of GAN-TTS training, we strongly recommend setting to false
cudnn_benchmark: false     # setting to true might acdelerate the training speed but sometimes decrease it
                           # therefore, we set to false as a default (recommend trying both cases)

EOF

cat <<'EOF' >> $ESPNET_ROOT/egs2/$1/tts1/all_run.sh
step=1

. ./utils/parse_options.sh

if [ $step -eq 1 ];then
. ./run.sh \
      --stage 1 \
      --stop-stage 5 \
      --g2p pyopenjtalk \
      --fs 44100 \
      --n_fft 2048 \
      --n_shift 512 \
      --dumpdir dump/44.1kHz \
      --win_length null \
      --tts_task gan_tts \
      --train_config ./conf/tuning/train_full_band_jets_gst_spk_harvest.yaml \
      --write_collected_feats true \
      --tts_stats_dir exp/full_band_jets_stats \
      --use_sid true \
      --min_wav_duration 0.75 \
      --f0max 1200 \
      --nj 8
step=2
echo -e "\n\n\nfinished feats extract"
fi

if [ ! -d downloads/full_band_jets_g2p_none_SEVEN ];then
  echo -e "\n\n\npretrained model directory is not found.\nDownloading it."
  [ ! -e "downloads" ] && mkdir -p "downloads"
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1t_KBgyvzjtw0BdwwBW4Txz_vkrozFI77' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1t_KBgyvzjtw0BdwwBW4Txz_vkrozFI77" -O "downloads/full_band_jets_g2p_none_SEVEN.zip" && rm -rf /tmp/cookies.txt
  unzip downloads/full_band_jets_g2p_none_SEVEN.zip -d downloads
  rm downloads/full_band_jets_g2p_none_SEVEN.zip
fi

if [ $step -eq 2 ];then
 
echo -e "\nfinished."
step=3
fi
 
 if [ $step -eq 3 ];then
   . ./run.sh \
    --stage 6 \
    --g2p pyopenjtalk \
    --fs 44100 \
    --n_fft 2048 \
    --n_shift 512 \
    --dumpdir dump/44.1kHz \
    --win_length null \
    --tts_task gan_tts \
    --write_collected_feats true \
    --tts_stats_dir exp/full_band_jets_stats \
    --use_sid true \
    --train_config ./conf/tuning/train_full_band_jets_gst_spk_harvest.yaml \
    # --train_args "--init_param exp/tts_train_jets_gst_spk_raw_phn_jaconv_pyopenjtalk_myg2p/348epoch.pth --ignore_init_mismatch True" \
    --tag train_jets_gst_spk
fi
EOF

cat <<'EOF' >> $ESPNET_ROOT/egs2/$1/tts1/inference.py
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import yaml

from espnet2.bin.tts_inference import Text2Speech
from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p
from espnet2.text.token_id_converter import TokenIDConverter

prosodic = False

parser = argparse.ArgumentParser()

parser.add_argument("--model_tag")
parser.add_argument("--train_config")
parser.add_argument("--model_file")
parser.add_argument("--vocoder_tag")
parser.add_argument("--vocoder_config")
parser.add_argument("--vocoder_file")
parser.add_argument("-p", "--prosodic",
                    help="Prosodic text input mode", action="store_true")
parser.add_argument(
    "--preg2p", help="preprocessing g2p use true or false", action="store_true")
parser.add_argument("--fs", type=int, default=24000)
parser.add_argument("--use_gst", action="store_true")
parser.add_argument("--use_sid", action="store_true")

args = parser.parse_args()

# Case 2: Load the local model and the pretrained vocoder
print("download model = ", args.model_tag, "\n")
print("download vocoder = ", args.vocoder_tag, "\n")
print("モデルを読み込んでいます...\n")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if args.model_tag is not None :
    text2speech = Text2Speech.from_pretrained(
        model_tag=args.model_tag,
        vocoder_tag=args.vocoder_tag,
        device=device,
    )
elif args.vocoder_tag is not None :
    text2speech = Text2Speech.from_pretrained(
        train_config=args.train_config,
        model_file=args.model_file,
        vocoder_tag=args.vocoder_tag,
        device=device,
    )
else :
    text2speech = Text2Speech.from_pretrained(
        train_config=args.train_config,
        model_file=args.model_file,
        # vocoder_config=args.vocoder_config,
        # vocoder_file=args.vocoder_file,
        device=device,
    )
with open(args.train_config) as f:
    config = yaml.safe_load(f)
config = argparse.Namespace(**config)

with open(config.train_data_path_and_name_and_type[1][0]) as f:
    lines = f.readlines()

guide = "セリフを入力してください"
if args.prosodic :
    guide = "アクセント句がスペースで区切られた韻律記号(^)付きのセリフをすべてひらがなで入力してください。(スペースや記号もすべて全角で)\n"
x = ""
while (1):
    # decide the input sentence by yourself
    print(guide)

    x = input()
    if x == "exit" :
        break
    if args.use_gst:
        print("wav.scpの行番号を入力してください")
        refer_id = input()
        path_str = lines[int(refer_id)].split(' ')[1].strip()
        speech, speech_fs = torchaudio.load(path_str)
        print(f'reference audio file: {path_str}')
        sid = None
    if args.use_sid:
        print("話者ラベルを指定してください")
        speaker_id = input()
        sid = torch.tensor(int(speaker_id)).unsqueeze(0)

    # model, train_args = TTSTask.build_model_from_file(
    #        args.train_config, args.model_file, "cuda"
    #        )

    token_id_converter = TokenIDConverter(
        token_list=text2speech.train_args.token_list,
        unk_symbol="<unk>",
    )

    if args.preg2p:
        token = pyopenjtalk_g2p(x)
        text_ints = token_id_converter.tokens2ids(token)
        text = np.array(text_ints)
    else:
        text = x

    # synthesis
    input_dict = {"text": text}
    if "use_gst" in config.tts_conf["generator_params"]:
        if config.tts_conf["generator_params"]["use_gst"]:
            input_dict["speech"] = speech[0]
            print(speech.shape)
    if args.use_sid:
        input_dict["sids"] = sid

    with torch.no_grad():
        start = time.time()
        data = text2speech(**input_dict)
        wav = data["wav"]
        # print(text2speech.preprocess_fn("<dummy>",dict(text=x))["text"])
    rtf = (time.time() - start) / (len(wav) / text2speech.fs)
    print(f"RTF = {rtf:5f}")

    if not os.path.isdir("generated_wav"):
        os.makedirs("generated_wav")

    if args.model_tag is not None :
        if "tacotron" in args.model_tag :
            mel = data['feat_gen_denorm'].cpu()
            plt.imshow(torch.t(mel).numpy(),
                       aspect='auto',
                       origin='lower',
                       interpolation='none',
                       cmap='viridis'
                       )
            plt.savefig('generated_wav/' + x + '.png')
    else :
        if "tacotron" in args.model_file :
            mel = data['feat_gen_denorm'].cpu()
            plt.imshow(torch.t(mel).numpy(),
                       aspect='auto',
                       origin='lower',
                       interpolation='none',
                       cmap='viridis'
                       )
            plt.savefig('generated_wav/' + x + '.png')
        if "fastspeech2" in args.model_file:
            print(data["pitch"].squeeze())

    # let us listen to generated samples
    import numpy as np
    from IPython.display import Audio, display

    #display(Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs))
    #Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs)
    np_wav = wav.view(-1).cpu().numpy()

    print("サンプリングレート", args.fs, "で出力します。")
    from scipy.io.wavfile import write
    samplerate = args.fs
    t = np.linspace(0., 1., samplerate)
    amplitude = np.iinfo(np.int16).max
    data = amplitude * np_wav / np.max(np.abs(np_wav))
    write("generated_wav/" + x + ".wav", samplerate, data.astype(np.int16))
    print("\n\n\n")

EOF

cat <<'EOF' >> $ESPNET_ROOT/egs2/$1/tts1/inference.sh
model_tag="kan-bayashi/jsut_tacotron2_accent_with_pause"
train_config=""
model_file=""

vocoder_tag=""
vocoder_config=""
vocoder_file=""

prosodic="false"
preg2p="false"
fs=""

use_gst=""
use_sid=""

. utils/parse_options.sh

COMMAND="python inference.py "


pwg=`pip list | grep parallel`
if [ "$pwg" == "" ];
then
	pip install -U parallel_wavegan
fi

ip=`pip list | grep ipython`
if [ "$pwg" == "" ];
then
	pip install -U IPython
fi


if [ "$train_config" == "" ] && [ "$model_file" == "" ]
then
	COMMAND="${COMMAND}--model_tag \"${model_tag}\" "
else
	COMMAND="${COMMAND}--train_config \"${train_config}\" "
	COMMAND="${COMMAND}--model_file \"${model_file}\" "
fi

if [ "$vocoder_config" == "" ] && [ "$vocoder_file" == "" ]
then
    if [ ! -z "$vocoder_tag" ];then
	    COMMAND="${COMMAND}--vocoder_tag \"${vocoder_tag}\" "
    fi
else
	COMMAND="${COMMAND}--vocoder_config \"${vocoder_config}\" "
	COMMAND="${COMMAND}--vocoder_file \"${vocoder_file}\" "
fi

if [ ! "$fs" == "" ]; then COMMAND="${COMMAND}--fs ${fs} "; fi

if [ "$prosodic" == "true" ]; then COMMAND="${COMMAND}-p "; fi
if [ "$preg2p" == "true" ]; then COMMAND="${COMMAND}--preg2p "; fi
if [ "$use_gst" == "true" ]; then COMMAND="${COMMAND}--use_gst "; fi
if [ "$use_sid" == "true" ]; then COMMAND="${COMMAND}--use_sid"; fi

echo "${COMMAND}"
echo ""
echo ""

eval $COMMAND

EOF


chmod +x $ESPNET_ROOT/egs2/$1/tts1/all_run.sh
chmod +x $ESPNET_ROOT/egs2/$1/tts1/inference.sh

cd $ESPNET_ROOT/egs2/$1/tts1/
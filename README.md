# Research Papers and Metrics

This repository contains an overview of research papers and their associated metrics.

| Link | Conference/Rank | Name | Title | Comparison | Description | Metrics | Repo |
|------|-----------------|------|-------|------------|-------------|---------|------|
| [Link](https://arxiv.org/abs/2401.08095) | nan | nan | DurFlex-EVC | nan | Emotion Conversion of the voice | nan | nan |
| [Link](https://www.isca-archive.org/interspeech_2024/huang24_interspeech.pdf) | nan | nan | DiffVC+ | nan | Ananymization using Gaussian sampled random style embeddings | nan | nan |
| [Link](http://www.arxiv.org/abs/2409.02245) | nan | nan | FastVoiceGrad | nan | Faster Diffusion-based VC using one-step inference in reverse diffusion | nan | nan |
| [Link](https://arxiv.org/abs/2406.09844) | nan | nan | Vec-Tok-VC+ | nan | Decomposing content and speaker features using RVQ and Clustering | nan | nan |
| [Link](https://arxiv.org/abs/2408.04708) | nan | nan | MulliVC | nan | Cross Language VC in absence of parallel cross lingual data | nan | nan |
| [Link](https://arxiv.org/abs/2311.04693) | ★★★★★ | Interspeech | Diff-HierVC: Diffusion-based Hierarchical Voice Conversion with Robust Pitch Generation and Masked Prior for Zero-shot Speaker Adaptation. | Have speaker loss, superivied training but results are decent | Voice2Voice: Two diffuison models DiffPitch and DiffVoice. DiffPitch for taget prompt and DiffVoice for final converted voice
Dataset: VCTK | CER
WER
EER
Speaker Similarity
MOS | https://github.com/hayeong0/Diff-HierVC |
| [Link](https://koe.ai/papers/llvc.pdf) | N.A | N.A. | Low-latency Real-time Voice Conversion on CPU | Any-to-one voice conversion, we are any-to-any | Voice2Voice: One generator, light model 
Dataset: LibriSpeech | nan | https://github.com/KoeAI/LLVC |
| [Link](https://arxiv.org/abs/2308.06382) | ★★ | AAAI | Phoneme Hallucinator: One-shot Voice Conversion via Set Expansion. | Low accuracy and fidelity | Voice2Voice: WavLM for source and target speech, Neighbour based VC pipeline 
Dataset: LibriSpeech | nan | https://github.com/PhonemeHallucinator/Phoneme_Hallucinator |
| [Link](https://arxiv.org/abs/2205.09784) | ★★★★★ | Interspeech | End-to-End Zero-Shot Voice Conversion with Location-Variable Convolutions | test results are suspicious (needs further testing), supervised training, have speaker loss | Voice2Voice:  ECAPA-TDNN for speaker embeddings. Content and speaker features from training utterances are used to reconstruct 
Dataset: VCTK | CER
EER
Speaker Similarity
MOS | https://github.com/wonjune-kang/lvc-vc |
| [Link](https://arxiv.org/pdf/2305.18975.pdf) | ★★★★★ | Interspeech | Voice Conversion With Just Nearest Neighbors. | Results on their demo are not correct, very bad accuracy and fidelity, does not preserve context correctly | Voice2Voice: Using WavLM and kNN regressor (self-supervised) 
Dataset: LibriSpeech 
 | CER
WER
EER
Speaker Similarity
MOS | https://github.com/bshall/knn-vc |
| [Link](https://arxiv.org/abs/2212.14227) | ★★★ | IEEE Spoken Language Technology Workshop (SLT) | Styletts-vc: One-shot voice conversion by knowledge transfer from style-based tts models. | Results are based on same utterance (for prompt and source), despite their supervised training they have bad accuracy, using text encoder and text aligner, they have been awareded in "SLT 2022 Best Papers Award"!!!! | Voice2Voice: Transfering the style of one speaker to another 
Dataset: VCTK 
 | nan | https://github.com/yl4579/StyleTTS-VC |
| [Link](https://arxiv.org/abs/2303.09057) | ★★★★★ | ICASSP  | TriAAN-VC: Triple Adaptive Attention Normalization for Any-to-Any Voice Conversion. | Results are suspecious, supervised training, evaluating only on 20 speakers (our model is on 1600) | Voice2Voice: comprises two encoders, extracting content and speaker information respectively, and a decoder. The encoders and decoder are connected via a bottleneck layer, similar approach to our work 
Dataset: VCTK  | CER
WER
Speaker Similarity
MOS | https://github.com/winddori2002/TriAAN-VC |
| [Link](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/2023%20SLT2022-Beta_VAE_based_one_shot_cross_lingual_VC.pdf) | ★★★ | IEEE Spoken Language Technology Workshop (SLT) | DISENTANGLED SPEECH REPRESENTATION LEARNING FOR ONE-SHOT CROSS-LINGUAL VOICE CONVERSION USING β-VAE | Supervised training, contents are not properly preserved | Voice2Voice:  one-shot cross-lingual voice conversion task to demonstrate the effectiveness of the disentanglement. Inspired by β-VAE
Dataset: VCTK | nan | https://github.com/light1726/BetaVAE_VC |
| [Link](https://arxiv.org/abs/2210.15418) | ★★★★★ | ICASSP  | FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion. | Supervised training, speaker loss and speaker encoder, minor problems with content, needs testing | Voice2Voice: FreeVC contains a prior encoder, a posterior encoder, a decoder, a discriminator and a speaker encoder
Dataset: VCTK
 | CER
WER
MOS
F0-PCC (Pearson correlation coefficient) | https://github.com/OlaWod/FreeVC |
| [Link](https://arxiv.org/abs/2203.16937) | ★★★★★ | Interspeech | HiFi-VC: High Quality ASR-Based Voice Conversion. | Not preserving context, does not work  on unseen speaker properly (Femal2Male), their demo is working good but when I tested on unseen speaker results were bad | Voice2Voice: HiFi-Gan as vocoder, ASR based content encoder, speaker embedder similar to NVC-Net
Dataset: VCTK | CER
WER
MOS
F0-PCC (Pearson correlation coefficient) | https://github.com/tinkoff-ai/hifi_vc |
| [Link](https://arxiv.org/abs/2209.04530) | ★★★★★ | Interspeech | DeID-VC: Speaker De-identification via Zero-shot Pseudo Voice Conversion. | Results are not good, not preserving style and linguistic, artificial, text dependent | Voice2Voice: a Variational Autoencoder (VAE) based Pseudo Speaker Generator (PSG) and a voice conversion Autoencoder (AE) under zero-shot settings
Dataset: WSJ | EER
WER
MOS | https://github.com/a43992899/DeID-VC |
| [Link](https://arxiv.org/pdf/2109.13821.pdf) | ★★★★★ | ICLR | Diffusion-based voice conversion with fast maximum likelihood sampling scheme. | Supervised training, results should be tested, current demo has bad results compared to ours | Voice2Voice: A one-shot many-to-many VC model. Introduces a novel DPM sampling scheme and establishes
its connection with likelihood maximization
Dataset: VCTK, LibriTTS | MOS | https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC |
| [Link](https://arxiv.org/abs/2106.00992) | ★★★★★ | ICASSP  | Nvc-net: End-to-end adversarial voice conversion. | Duration of their test samples are low (suspecious), needs further testing, supervised training | Voice2Voice: An ASR model is used to extract the linguistic features, speaker encoder extracts speaker embeddings
Dataset: VCTK | MOS | https://github.com/sony/ai-research-code/tree/master/nvcnet |
| [Link](https://arxiv.org/abs/2009.02725v3) | Journal | IEEE/ACM Transactions on Audio, Speech, and Language Processing | Any-to-many voice conversion with location-relative sequence-to-sequence modeling | Text supervision, results not good, fidelity low | Voice2Voice: an encoder-decoder-based (CTC-attention) phoneme recognizer for content. Multi-speaker location-relative attention
based seq2seq synthesis model for speaker identity 
Dataset: VCTK | nan | https://github.com/liusongxiang/ppg-vc |
| [Link](https://arxiv.org/abs/2302.08296) | ★★★★ | IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) | QuickVC: Any-to-many Voice Conversion Using Inverse Short-time Fourier Transform for Faster Conversion | Results good on their ow data, poor accuracy on unseen speakers (pulished in Dec 2023) | Voice2Voice: MS-iSTFT-Decoder vocoder, HuBERT encoder, Speaker encoder is one LSTM layer, relavant to our work
Dataset: VCTK
 | MOS
CER
WER | https://github.com/quickvc/QuickVC-VoiceConversion |
| [Link](https://arxiv.org/pdf/2104.02901v2.pdf) | ★★★★★ | Interspeech | S2VC: A Framework for Any-to-Any Voice Conversion with Self-Supervised Pretrained Representations | Low accuracy, low fidellity | Voice2Voice: The overall framework evolves from FragmentVC. Self-attention pooling guides the representation encoded by the
source encoder to be close to that encoded by the target encoder.
Dataset: VCTK | MOS | https://github.com/howard1337/S2VC |
| [Link](https://arxiv.org/pdf/2104.00931v2.pdf) | ★★★★★ | ICASSP  | Assem-VC: Realistic Voice Conversion by Assembling Modern Speech Synthesis Techniques | Text2Speech Method | TextMel2Speech: Input source text and target mel generates converted mel and using fine-tuned HiFiGAN converted to voice | nan | https://github.com/mindslab-ai/assem-vc |
| [Link](https://arxiv.org/pdf/2104.00355v3.pdf) | ★★★★★ | Interspeech | Speech Resynthesis from Discrete Disentangled Self-Supervised Representations | Pitch2Unit, Supervised Speaker encoder | Voice2Voice: Hubert (Speech2unit)+Pitch2Unit+SPeaker EMbedder (Speaker ID pre-trained) + HiFiGAN Vocoder | PER
WER
MOS
EER | https://github.com/facebookresearch/speech-resynthesis |
| [Link](https://arxiv.org/pdf/2103.16809v2.pdf) | ★★★★★ | Interspeech | Limited Data Emotional Voice Conversion Leveraging Text-to-Speech: Two-stage Sequence-to-Sequence Training | Text2Speech Method | Text2Speech: 5 components, text enc, seq2seq ASR enc, style enc (emotion enc), classifier (emotion), se2seq decoder. Two staged training.  | nan | https://github.com/KunZhou9646/seq2seq-EVC |
| [Link](https://arxiv.org/pdf/2102.12841v1.pdf) | ★★★★★ | ICASSP  | MaskCycleGAN-VC: Learning Non-parallel Voice Conversion with Filling in Frames | Single Generator, filling masked frames, totally different work than ours | Voice2Voice: Modified version of CycleGAN VC-2 trained using filling in frames (or in mel-spectrum) similar to image inpainting.  | nan | https://github.com/GANtastic3/MaskCycleGAN-VC |
| [Link](https://arxiv.org/pdf/2011.05731v2.pdf) | ★★★★ | ICME | FastSVC: Fast Cross-Domain Singing Voice Conversion with Feature-wise Linear Modulation | Different Generator, No seperate Speaker Encoder | Voice2Voice:Singing Voice Conversion, light weight, Conformer-based phoneme recognizer that extracts singer-agnostic linguistic features from singing signals.  | nan | https://github.com/liusongxiang/ppg-vc |
| [Link](https://arxiv.org/pdf/2010.14150v2.pdf) | ★★★★★ | ICASSP  | FragmentVC: Any-to-Any Voice Conversion by End-to-End Extracting and Fusing Fine-Grained Voice Fragments With Attention | Target Encoder based on log-mel, Custom Decoder and Encoder Blocks | Voice2Voice: Wav2Vec2 for content (source), log mel-spectogram for spectral features (target). Two-stged training for feature aligning based on Transformers. Reconstruction loss. | EER
MOS | https://github.com/yistLin/FragmentVC |
| [Link](https://arxiv.org/pdf/2010.11672v1.pdf) | ★★★★★ | Interspeech | CycleGAN-VC3: Examining and Improving CycleGAN-VCs for Mel-spectrogram Conversion | Really Heavy Model (Heavier CycleGAN) with TFAN, Poor Results on Unseen | Voice2Voice: Non-parallel VC challenge, CycleGAN-VC limitations, CycleGAN-VC3 with TFAN ( time-frequency adaptive normalization) enhancement, preserves time-frequency structure, outperforms CycleGAN-VC2 in naturalness and similarity. (see MaskCycleGAN-VC) | nan | https://github.com/jackaduma/CycleGAN-VC3 |
| [Link](https://arxiv.org/pdf/2010.08136v1.pdf) | ★★★★★ | Interspeech | Towards Natural Bilingual and Code-Switched Speech Synthesis Based on Mix of Monolingual Recordings and Cross-Lingual Voice Conversion | Text2Speech Method | Text2Voice: Aim to build a good bilingual speech synthesis system, with native like output. Based on Tacotron-2 and a Transformer-based synthesizer for augmentation to get bilingual data. | nan | https://github.com/espnet/espnet |
| [Link](https://arxiv.org/pdf/2010.02434v1.pdf) | N.A | Joint Workshop for the Blizzard Challenge and Voice Conversion Challenge  | The Sequence-to-Sequence Baseline for the Voice Conversion Challenge 2020: Cascading ASR and TTS | Used supervised ASR inside, not Unsupervised, Leads to poor results | Voice2Voice: A naive approach converting voice to text using ASR followed by a TTS model plus a Neural Vocoder on top. | nan | https://github.com/espnet/espnet |
| [Link](https://arxiv.org/pdf/2008.12604v7.pdf) | Journal | IEEE/ACM Transactions on Audio, Speech, and Language Processing | Nonparallel Voice Conversion with Augmented Classifier Star Generative Adversarial Networks | Learning different speakers using augmentation, poor results, robotic voice | Mel2Mel: StarGAN-based proposing "augmented classifier StarGAN" that does not require any info about the domain of the speech at test time. USing a single generator, it learns the mapping among multiple speech domains. | nan | https://github.com/kamepong/StarGAN-VC |
| [Link](https://arxiv.org/pdf/2006.04154v1.pdf) | ★★★★★ | Interspeech | VQVC+: One-Shot Voice Conversion by Vector Quantization and U-Net architecture | Focused on single enc single dec, and quantization, trained on VCTK | Voice2Voice: Feature disentanglement using U-Net based neural network and vector quantization.(Using skip connections and quantized connections) | nan | https://github.com/ericwudayi/SkipVQVC |
| [Link](https://arxiv.org/pdf/1806.02169v2.pdf) | ★★★ | IEEE Spoken Language Technology Workshop (SLT) | StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks | many-to-many, no seperate content and speaker enc, poor results | Voice2Voice: While CycleGAN-VC allows the generation of naturalsounding speech when a sufficient number of training examples are available, one limitation is that it only learns one-to-one-mappings. Here, we propose using StarGAN [42] to develop a method that allows non-parallel many-to-many VC. We call the present method StarGAN-VC. (Highly cited) | nan | https://github.com/kamepong/StarGAN-VC |
| [Link](https://arxiv.org/pdf/1905.05879v2.pdf) | ★★★★★ | International Conference on Machine Learning | AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss | Really old and poor results, simple style transfer technique | Voice2Voice: Autoencoder-based style transfer using two encoders for source, and target and a decoder for generation. | nan | https://github.com/liusongxiang/StarGAN-Voice-Conversion |
| [Link](https://arxiv.org/pdf/1904.05742v4.pdf) | ★★★★★ | Interspeech | One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization | VAE based, bad results | Voice2Voice: Disentangles speaker and content representations using Instance Normalization (IN). (Claim) This allows their model to do VC for unseen speakers during the training. They use, Speaker encoder, Content encoder (with IN),  and a decoder (with AdaIN) to do the VC | nan | https://github.com/jjery2243542/adaptive_voice_conversion |
| [Link](https://arxiv.org/pdf/2401.08095v1.pdf) | N.A | N.A. | DurFlex-EVC: Duration-Flexible Emotional Voice Conversion with Parallel Generation | Supervised emotion training | Voice2Voice: By intergrating a style auto-encoder and unit aligner, and training using parallel dataset of ESD (for emotion) they calim more control over Emotional VC. Style autoencoder is used for disentanglement and manipulation of style elements. (Pretty complicated stuff here) | nan | https://github.com/hs-oh-prml/durflexevc |
| [Link](https://arxiv.org/pdf/2309.07586v1.pdf) | ★★★★★ | Interspeech | Emo-StarGAN: A Semi-Supervised Any-to-Many Non-Parallel Emotion-Preserving Voice Conversion | Supervised emotion training | Voice2Voice: A modifed STarGAN-VC version trained using emotion labelled and non-parallel datasets plus integrating emotion loss componnents to better preserve emotion. (Basically used StarGAN-VC while adding some loss terms) | PCC
CER
WER
MOS | https://github.com/suhitaghosh10/emo-stargan |
| [Link](https://arxiv.org/pdf/2306.10588v1.pdf) | ★★★★★ | Interspeech | DuTa-VC: A Duration-aware Typical-to-atypical Voice Conversion Approach with Diffusion Probabilistic Model | Diffusion based, multiple mel-loss, bad results | Voice2Voice: Encoder (phoneme and phoneme duration predictor) for source mel, Decoder for reverse diffusion (Diffusion probab. models) leading to generating target mel, and a Vocoder to recosntruct waveform. | WER
EER
MOS | https://github.com/wanghelin1997/duta-vc |
| [Link](https://arxiv.org/pdf/2209.11866v5.pdf) | ★★★★★ | Interspeech | ControlVC: Zero-Shot Voice Conversion with Time-Varying Controls on Pitch and Speed | Has pitch encoder, Speaker enc is an LSTM later with FC, Competetive results | Voice2Voice: Time varying control on pitch and speed. Modifying the speed of
the source utterance using TD-PSOLA pre-processing, pitch control by modifying the pitch contour of the speed-controlled source utterance, and uses a VQ-VAE pitch encoder to compute discrete pitch embedding, HuBERT extracts the linguistic embedding, and A modified version of the HiFi-GAN vocoder to generate waveform. | WER
EER
MOS | https://github.com/MelissaChen15/control-vc |
| [Link](https://arxiv.org/pdf/2205.09784v3.pdf) | ★★★★★ | Interspeech | End-to-End Zero-Shot Voice Conversion with Location-Variable Convolutions | E2E small model | Voice2Voice: (MIT) They have employed a kernel predictor for LVC inside their Generator which consists of multiple LVC blocks and do VC by using F0 extraction for content info.They claim this is the first end2end pipeline with good quality and zero-shot capability | MOS
EER
CER
WER | https://github.com/wonjune-kang/lvc-vc |
| [Link](https://arxiv.org/pdf/2205.05227v2.pdf) | ★★★★★ | Interspeech | Towards Improved Zero-shot Voice Conversion with Conditional DSVAE | Parallel training and supervised | Voice2Voice: Conditioning on content bias using proposed disentangled
sequential variational autoencoder (DSVAE).  Trained on VCTK | MOS | https://github.com/jlian2/Improved-Voice-Conversion-with-Conditional-DSVAE |
| [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9262021&tag=1) | ★★★★★ | nan | An Overview of Voice Conversion and Its Challenges: From Statistical Modeling to Deep Learning | Overview of various VC model types | Overview from statitical models to DL methods | nan | nan |
| [Link](https://arxiv.org/pdf/2310.09653.pdf) | N.A | nan | SELFVC: VOICE CONVERSION WITH ITERATIVE REFINEMENT USING SELF TRANSFORMATIONS | SelfVC-Nvidia/UCSD | nan | CER
PER
GPE
EER
MOS
 | nan |
| [Link](https://arxiv.org/pdf/2203.14156v1.pdf) | ★★★★★ | ICASSP  | SpeechSplit 2.0: Unsupervised speech disentanglement for voice conversion Without tuning autoencoder Bottlenecks | nan | Voice2Voice: Speech component disentangled at the autoencoder input using efficient signal processing techniques instead of relying on bottleneck tuning. | nan | https://github.com/biggytruck/speechsplit2 |
| [Link](https://arxiv.org/pdf/2112.02418v4.pdf) | ★★★★★ | ICML | YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone | nan | Text2Voice: multilingual transformer-based text encoder, stochastic duration prediction and speaker consistency loss during training for improved synthesis quality and speaker similarity. HiFi-GAN for waveform generation. Inference: the model utilizes Monotonic Alignment Search for generating human-like rhythms of speech  | MOS
EER | https://github.com/coqui-ai/TTS |
| [Link](https://arxiv.org/pdf/2111.03811v3.pdf) | ★★★★★ | ICASSP  | SIG-VC: A Speaker Information Guided Zero-shot Voice Conversion System for Both Human Beings and Machines | nan | Voice2Voice: Supervise the intermediate representation to remove the speaker information from the linguistic information, pre-trained acoustic model used to extract the linguistic feature | MOS
EER | https://github.com/HaydenCaffrey/SIG-VC |
| [Link](https://arxiv.org/pdf/2110.06280v1.pdf) | ★★★★★ | ICASSP  | S3PRL-VC: Open-source Voice Conversion Framework with Self-supervised Speech Representations | nan | Voice2Voice: The source speech are first extracted by a recognizer, then synthesized into the converted speech by a synthesizer, then speech representations is learned from large-scale unlabeled data | nan | https://github.com/s3prl/s3prl |
| [Link](https://arxiv.org/pdf/2107.10394v2.pdf) | ★★★★★ | Interspeech | StarGANv2-VC: A Diverse, Unsupervised, Non-parallel Framework for Natural-Sounding Voice Conversion | nan | VoiceMel2Speech: generator converting mel-spectrograms into frequency representations, a mapping network for diverse style generation, a style encoder for reference-based style extraction, and discriminators to capture domain-specific features, including an additional classifier for feedback on domain-specific characteristics to improve sample similarityM | MOS
CER
EER | https://github.com/yl4579/StarGANv2-VC |
| [Link](https://arxiv.org/pdf/2106.10132v1.pdf) | ★★★★★ | Interspeech | VQMIVC: Vector Quantization and Mutual Information-Based Unsupervised Speech Representation Disentanglement for One-shot Voice Conversion | nan | VoiceMel2Voice: Content encoder, speaker encoder, pitch extractor, and decoder for unsupervised voice conversion. It utilizes variational approximation networks to minimize mutual information (MI) between content, speaker, and pitch representations, achieving disentanglement without text or speaker labels. One-shot voice conversion is achieved by extracting representations from the source and target speakers, then using the decoder to generate converted mel-spectrograms. | MOS
CER
WER
F0-PCC | https://github.com/Wendison/VQMIVC |

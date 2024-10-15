# Honey, I Shrunk the ML Model

## Definition of the Project

Our goal with this project is to shrink an existing Speech-to-Text machine learning (ML) model down to run on as small a device as possible. We will be using the TinyML framework to achieve this. Ideally, this model will be small enough to run on a wearable device, such as a smartwatch or a pair of smart glasses. The model will be trained on a dataset of speech samples and will be able to convert spoken words into text. The model will be evaluated based on its accuracy, speed, and size.

For members of the deaf community who need a quick and reliable way to understand spoken language in real-time, who want an accessible, portable, and affordable solution for speech-to-text translation, the Speech-to-Text Translator is a compact, wearable device that uses a highly optimized TinyML model to convert spoken language into text instantly, displayed on a small screen or mobile device, unlike existing expensive or bulky devices that rely on cloud-based processing or are slow in translating speech, our product provides real-time, accurate translation locally on a low-cost microcontroller or single-board computer (SBC), ensuring privacy, portability, and affordability for everyday use.

## Target Users

The target users for this project are people who need to convert spoken words into text on a wearable device. This could include people who are deaf or hard-of-hearing, need to take notes on the go, need to send text messages while driving, or need to communicate with others in noisy environments.

## User Stories

1. **As a deaf user**, I want to use the wearable device to convert spoken words into text so that I can understand conversations in real-time.
1. **As a student**, I want to use the wearable device to take notes during lectures by converting the professor's speech into text, so I can focus on listening and understanding the material.
1. **As a driver**, I want to use the wearable device to send text messages by speaking, so I can communicate without taking my hands off the wheel.
1. **As a professional in a noisy environment**, I want to use the wearable device to convert spoken instructions into text, so I can ensure I don't miss any important information.
1. **As a person with a speech disability**, I want to use the wearable device to convert my spoken words into text, so I can communicate more effectively with others.

## Minimum Value Product (MVP)

The MVP for this project is a speech-to-text model that can run on a SBC. The model should be able to convert spoken words into text with high accuracy, low latency, and a small memory footprint. The model should be trained on a dataset of speech samples and should be able to recognize a wide range of words and phrases. The model should be evaluated based on its accuracy, speed, and size.

## Literature Review

### TinyML for Speech Recognition

TinyML, which refers to machine learning models that can run on resource-constrained edge devices, has shown promising results for speech recognition tasks. Several recent papers have explored applying TinyML techniques to enable speech-to-text capabilities on low-power microcontrollers and embedded systems. Banbury et al. (2021) presented an overview of the emerging field of TinyML, highlighting its potential for enabling on-device speech recognition. The authors discuss how techniques like model compression and quantization allow deploying neural networks on microcontrollers with limited memory and compute

- Wong et al. (2020) proposed TinySpeech, a novel neural architecture for keyword spotting on microcontrollers. Their model uses attention condensers to achieve high accuracy while maintaining a small footprint suitable for edge devices. Experiments showed TinySpeech outperforming larger models on several keyword spotting benchmarks
- Lin et al. (2020) developed MCUNet, a framework for designing compact neural networks that can run inference on microcontrollers. They demonstrated MCUNet's effectiveness for speech commands recognition, achieving 96% accuracy on the Google Speech Commands dataset while fitting within 256KB of memory
- Fedorov et al. (2021) introduced TinyLSTMs, an efficient implementation of LSTM networks for speech enhancement on hearing aid devices. By leveraging fixed-point quantization and model compression, they were able to deploy real-time speech enhancement on an ultra-low power DSP chip
- Sudharsan et al. (2021) proposed an end-to-end TinyML pipeline for deploying personalized speech recognition models on edge devices. Their approach uses transfer learning and pruning to create compact, user-specific models that can run on microcontrollers

### Conclusion

These papers demonstrate the significant progress in enabling speech recognition capabilities on highly resource-constrained devices through TinyML techniques. Key approaches include model architecture innovations, quantization, pruning, and hardware-aware neural architecture search. While challenges remain, TinyML is opening up new possibilities for ubiquitous speech interfaces across IoT devices.

## Sprint Breakdown and Objectives

We have structured our project into several sprints to ensure a focused and organized approach to development. Each sprint has specific goals to help us gradually build towards the final product.

### Sprint 1: Research and Exploration

The first sprint was dedicated to conducting thorough research on the topics of TinyML, Speech-to-Text models, and relevant optimization techniques. During this sprint, we explored recent advancements in TinyML, particularly in speech recognition applications. This research provided the foundation for our technical approach and narrowed down potential frameworks and techniques, including model compression, quantization, and pruning.

### Sprint 2: Model Selection and Dataset Identification

In Sprint 2, we chose the Whisper model as our base speech-to-text model, recognizing its high accuracy and efficiency. We also identified relevant datasets for training and testing, ensuring that they reflect the wide range of real-world speech patterns. The Google Speech Commands dataset and other publicly available speech datasets were shortlisted, which would help us refine the model's ability to recognize spoken language across different environments and accents.

### Sprint 3: Model Training and Optimization

Sprint 3 focuses on training the Whisper model and optimizing it for edge deployment. We will apply techniques such as pruning and quantization to shrink the model's size while maintaining accuracy and performance. If necessary, we will also consider model distillation as a fallback strategy, further reducing the model's complexity. The goal is to create a highly efficient and compact version of the speech-to-text model that can run on resource-constrained devices.

### Sprint 4: Hardware Implementation

The final sprint will involve deploying the optimized model onto a low-cost microcontroller or single-board computer (SBC). We will work on integrating the model with the hardware, ensuring it runs smoothly in real-time with minimal latency. This step will also involve testing the performance of the model on the target device and making any necessary adjustments to ensure it meets the requirements of portability, speed, and accuracy.

## Citations of Papers

1. Tsoukas, V., Gkogkidis, A., Boumpa, E., & Kakarountas, A. (2024). A review on the emerging technology of TinyML. ACM Computing Surveys, 56(10), Article 259. <https://doi.org/10.1145/3661820>
1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. OpenAI. <https://github.com/openai/whisper>
1. Banbury, C. R., Reddi, V. J., Lam, M., Fu, W., Fazel, A., Holleman, J., ... & Whatmough, P. N. (2021). Benchmarking TinyML systems: Challenges and direction. arXiv preprint arXiv:2003.04821. <https://doi.org/10.48550/arXiv.2003.04821>
1. Wong, A., Famouri, M., Pavlova, M., Surana, S. (2020). TinySpeech: Attention condensers for deep speech recognition neural networks on edge devices. arXiv preprint arXiv:2008.04245.
1. Lin, J., Chen, W. M., Lin, Y., Cohn, J., Gan, C., & Han, S. (2020). MCUNet: Tiny deep learning on IoT devices. Advances in Neural Information Processing Systems, 33, 11711-11722. <https://doi.org/10.48550/arXiv.2007.10319>
1. Fedorov, I., Stamenovic, M., Jensen, C., Yang, L. C., Mandell, A., Gan, Y., ... & Mattina, M. (2021). TinyLSTMs: Efficient neural speech enhancement for hearing aids. arXiv preprint arXiv:2005.11138. <https://doi.org/10.48550/arXiv.2005.11138>
1. Sudharsan, B., Breslin, J. G., & Ali, M. I. (2021). Edge2Train: A framework to train machine learning models (SVMs) on resource-constrained IoT edge devices. Internet of Things, 14, 100187. <http://dx.doi.org/10.1145/3410992.3411014>

## AI Attributions

Written with the aid of GitHub Copilot (document ).
Literature review conducted with the aid of Perplexity Pro. Prompt: "Conduct a literature review of TinyML speech to text scientific papers. Provide APA references to each paper you review."

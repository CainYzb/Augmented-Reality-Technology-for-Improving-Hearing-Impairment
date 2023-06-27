#paddlespeech_server start --config_file C:\Users\17845\Desktop\Code\SoftwareEngineer\PaddleSpeech-r1.4.1\demos\speech_server\conf\application.yaml

from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor

asrclient_executor = ASRClientExecutor()
res = asrclient_executor(
    input="zh.wav",
    server_ip="127.0.0.1",
    port=9876,
    sample_rate=16000,
    lang="zh_cn",
    audio_format="wav")

print(res)
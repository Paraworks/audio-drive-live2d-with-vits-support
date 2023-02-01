import argparse
parser = argparse.ArgumentParser(
    description=
    '一些必须的文件')
parser.add_argument('--savf',
                    type=str,
                    help='存储谈话的文件.',
                    default='res/conversation.txt')
args = parser.parse_args()


while True:
    tts_text = input(":")
    text = '[ZH]' + tts_text + '[ZH]'
    with open(args.savf, "w", encoding="utf-8") as f1:
        f1.write(text)
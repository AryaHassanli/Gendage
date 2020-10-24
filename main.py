import os

from config import config
from helpers.demo import demo
from helpers.parseArguments import parseArguments

# Handle arguments
args = parseArguments('main')
if args.gradient:
    config.set(datasetDir='/storage/datasets',
               outputDir='/artifacts')
else:
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir='output')



def main():
    demo(lambda x: ['hi'],
         os.path.join(config.baseDir, 'inputSamples/Recording.mp4'),
         os.path.join(config.outputDir, 'result.avi'),
         2,
         config.device)


if __name__ == '__main__':
    main()

from multiprocessing import Process

def test_segment():
    from app import segmentation
    segmentation.main("..\\UBIRIS_200_150\\CLASSES_400_300_Part1")

def test_sr():
    from app import upscale
    upscale.main("test\\upscale\\hr")

def test_classifier():
    from app import classifier
    classifier.main("test\\upscale\\sr")


def test():
    p1 = Process(target=test_segment)
    p1.start()
    p1.join()

    p2 = Process(target=test_sr)
    p2.start()
    p2.join()

    p3 = Process(target=test_classifier)
    p3.start()
    p3.join()

if __name__ == '__main__':
    test()

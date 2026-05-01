# Changelog

## 2026-05-01

- Vosstanovleny baseline-dannye dlya `IRKUTSK-pseudolabel-unet-r50-t512-v4` i `GPU-deforest-segformer-b0-t1024-gpu-v13`.
- Dobavleny lokalnye baseline-artefakty v `artifacts/baselines`.
- V train loop vklyucheny realnye augmentatsii iz `train.augmentations`: flips, rot90, brightness_contrast/color_jitter, gamma, noise, blur, cutout/coarse_dropout.
- Dobavlena podderzhka `pseudolabel.images_uri`, chtoby trenirovat model na deforest dataset, a itogovuyu psevdo-razmetku strogo stroitt po `images/kanopus/irkutsk/`.
- Provereny lokalnye komandy `compileall` i `unittest`.

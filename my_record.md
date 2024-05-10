python test_pic.py --mode all --opt VDSR/tools/process_img2/options/option_set_03_vdsrwithsegJiekou/options.yml 

find . -name ".ipynb_checkpoints" -print
rm -rf .ipynb_checkpoints

zip -r originGT.zip ./originGT

VDSR: set 01 02 06
EDSR: set 03 04 05
RCAN: set 07

[['./test_4/0.png', [[203, 22, 505, 404]]], 
['./test_4/1.png', [[122, 44, 161, 101]]], 
['./test_4/2.png', [[44, 286, 104, 387]]], 
['./test_4/3.png', [[176, 41, 403, 320]]], 
['./test_4/4.png', []]]

[[<PIL.Image.Image image mode=RGB size=2028x1620 at 0x1CAB9C45C10>, <PIL.Image.Image image mode=L size=2028x1620 at 0x1CABFBF1A90>, <PIL.Image.Image image mode=L size=2028x1620 at 0x1CABFBF1A60>, [[115, 160, 1037, 1618]], [[115, 160, 1037, 1618]]], 
[None, None, None, [], []], 
[None, None, None, [], []], 
[None, None, None, [], []], 
[<PIL.Image.Image image mode=RGB size=1852x1288 at 0x1CAB9C455E0>, <PIL.Image.Image image mode=L size=1852x1288 at 0x1CAB9C45760>, <PIL.Image.Image image mode=L size=1852x1288 at 0x1CAB9C453A0>, [[209, 141, 818, 1147]], [[209, 141, 818, 1147]]]]

blend 0.png people...
psnr blend: 35.951964685586255
psnr total: 34.29207901667902

blend 1.png
psnr blend: 34.29489947332807
psnr total: 32.32979016293996

blend 2.png
psnr blend: 38.140856681592446
psnr total: 38.38306430777062

blend 3.png
psnr blend: 39.32293107574739
psnr total: 37.62596533401373

blend 4.png
psnr blend: 41.21030706565656
psnr total: 40.832064698005325


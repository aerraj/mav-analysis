"""Export the exact notebook checkpoint to two browser inference graphs."""
import sys, pathlib, hashlib, json
import torch, numpy as np, onnxruntime as ort
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
from magnet import MagNet
checkpoint = pathlib.Path(sys.argv[1])
model = MagNet().eval()
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict({k.replace('module.', ''): v for k,v in state.items()})
class Render(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.manipulator = model.manipulator; self.decoder = model.decoder
    def forward(self, texture, motion_a, motion_b, amplification):
        return self.decoder(texture, self.manipulator(motion_a, motion_b, amplification))
render = Render().eval()
a, b = torch.rand(1,3,64,96)*2-1, torch.rand(1,3,64,96)*2-1
with torch.no_grad():
    ta, ma = model.encoder(a); tb, mb = model.encoder(b)
    amp = torch.tensor([10.]).reshape(1,1,1,1)
out = ROOT/'public/models'
common = dict(opset_version=17, dynamo=False)
torch.onnx.export(model.encoder, (a,), str(out/'encoder.onnx'), input_names=['frame'], output_names=['texture','motion'], dynamic_axes={'frame':{2:'h',3:'w'}, 'texture':{2:'th',3:'tw'}, 'motion':{2:'mh',3:'mw'}}, **common)
torch.onnx.export(render, (tb,ma,mb,amp), str(out/'render.onnx'), input_names=['texture','motion_a','motion_b','amplification'], output_names=['frame'], dynamic_axes={'texture':{2:'th',3:'tw'},'motion_a':{2:'mh',3:'mw'},'motion_b':{2:'mh',3:'mw'},'frame':{2:'h',3:'w'}}, **common)
enc = ort.InferenceSession(str(out/'encoder.onnx')); dec = ort.InferenceSession(str(out/'render.onnx'))
errors=[]
for h,w in [(64,96),(96,128)]:
    a,b = torch.rand(1,3,h,w)*2-1,torch.rand(1,3,h,w)*2-1
    ea=enc.run(None,{'frame':a.numpy()}); eb=enc.run(None,{'frame':b.numpy()})
    for factor in [10,25,50]:
        amp=torch.tensor([float(factor)]).reshape(1,1,1,1)
        actual=dec.run(None,{'texture':eb[0],'motion_a':ea[1],'motion_b':eb[1],'amplification':amp.numpy()})[0]
        with torch.no_grad(): expected=model(a,b,0,0,amp,mode='evaluate').numpy()
        error=float(np.max(np.abs(actual-expected))); errors.append(error)
        assert np.allclose(actual,expected,atol=0.0002,rtol=0.001),error
meta={'checkpoint':'magnet_epoch12_loss7.28e-02.pth','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'parameters':sum(p.numel() for p in model.parameters()),'source_commit':'be33c252f670bcb55d1442e30491584f9564a22d','opset':17,'max_validation_error':max(errors),'models':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.onnx')}}
(out/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))

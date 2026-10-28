/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web';
import { Input, ALL_FORMATS, BlobSource, CanvasSink, CanvasSource, Output, Mp4OutputFormat, WebMOutputFormat, BufferTarget, canEncodeVideo } from 'mediabunny';
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModule from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {wasm: wasmUrl, mjs: wasmModule};
const send = (data: object) => self.postMessage(data);
self.onmessage = async ({data}: MessageEvent<{file:File;factors:number[];maxSize:number}>) => {
 let input: Input | undefined;
 const outputs: Output[] = [];
 let encoder: ort.InferenceSession | undefined, renderer: ort.InferenceSession | undefined;
 try {
  if (typeof VideoEncoder === 'undefined' || typeof VideoDecoder === 'undefined') throw new Error('This browser does not support video processing. Open MAV Lab in a recent Chrome or Edge browser.');
  send({type:'progress',percent:0,message:'Reading video and loading MagNet weights…'});
  input = new Input({formats:ALL_FORMATS,source:new BlobSource(data.file)});
  const track = await input.getPrimaryVideoTrack();
  if (!track || !await track.canDecode()) throw new Error('This video codec cannot be decoded by your browser. Try an H.264 MP4 file.');
  const duration = await input.computeDuration();
  const scale = Math.min(1, data.maxSize / Math.max(track.displayWidth,track.displayHeight));
  const width = Math.max(16, Math.round(track.displayWidth*scale/4)*4);
  const height = Math.max(16, Math.round(track.displayHeight*scale/4)*4);
  const mp4 = await canEncodeVideo('avc',{width,height});
  if (!mp4 && !await canEncodeVideo('vp9',{width,height})) throw new Error('No compatible video encoder found. Use a recent Chrome or Edge browser.');
  encoder = await ort.InferenceSession.create('/models/encoder.onnx',{executionProviders:['wasm']});
  renderer = await ort.InferenceSession.create('/models/render.onnx',{executionProviders:['wasm']});
  const canvas = new OffscreenCanvas(width,height);
  const ctx = canvas.getContext('2d',{willReadFrequently:true})!;
  const writers = data.factors.map(factor=> {
   const target = new BufferTarget();
   const output = new Output({format:mp4?new Mp4OutputFormat({fastStart:'in-memory'}):new WebMOutputFormat(),target});
   outputs.push(output);
   const source = new CanvasSource(canvas,{codec:mp4?'avc':'vp9',bitrate:3_000_000});
   output.addVideoTrack(source); return {factor,target,output,source};
  });
  for (const writer of writers) await writer.output.start();
  const sink = new CanvasSink(track,{width,height,fit:'fill',poolSize:2});
  let previous: ort.Tensor | null = null, frameCount = 0, firstTime = 0;
  const size = width*height;
  const started = performance.now();
  send({type:'metadata',width,height,duration,format:mp4?'MP4':'WebM'});
  for await (const frame of sink.canvases()) {
   if (!frameCount) firstTime = frame.timestamp;
   const timestamp = frame.timestamp - firstTime;
   ctx.drawImage(frame.canvas,0,0);
   const rgba = ctx.getImageData(0,0,width,height).data;
   const rgb = new Float32Array(size*3);
   for(let i=0;i<size;i++) {rgb[i]=rgba[i*4]/127.5-1;rgb[size+i]=rgba[i*4+1]/127.5-1;rgb[size*2+i]=rgba[i*4+2]/127.5-1;}
   const tensor = new ort.Tensor('float32',rgb,[1,3,height,width]);
   const features = await encoder.run({frame:tensor}); tensor.dispose();
   for (const writer of writers) {
    if (previous) {
     const amp = new ort.Tensor('float32',Float32Array.of(writer.factor),[1,1,1,1]);
     const result = await renderer.run({texture:features.texture,motion_a:previous,motion_b:features.motion,amplification:amp}); amp.dispose();
     const pixels = result.frame.data as Float32Array;
     const image = ctx.createImageData(width,height);
     for (let i=0;i<size;i++) {for(let c=0;c<3;c++)image.data[i*4+c]=Math.round((Math.max(-1,Math.min(1,pixels[c*size+i]))+1)*127.5);image.data[i*4+3]=255;}
     ctx.putImageData(image,0,0);result.frame.dispose();
    } else ctx.drawImage(frame.canvas,0,0);
    await writer.source.add(timestamp,frame.duration);
   }
   previous?.dispose();previous=features.motion;features.texture.dispose();frameCount++;
   const percent=Math.min(99,Math.round((timestamp+frame.duration)/duration*100));
   const elapsed=(performance.now()-started)/1000;
   send({type:'progress',percent,message:`Processing frame ${frameCount} · ${width} × ${height}`,frames:frameCount,elapsed});
  }
  previous?.dispose();
  if(frameCount<2) throw new Error('The video needs at least two decodable frames.');
  send({type:'progress',percent:99,message:'Finalizing downloadable videos…'});
  for(const w of writers){w.source.close(); await w.output.finalize();}
  const results=writers.map(w=>({factor:w.factor,blob:new Blob([w.target.buffer!],{type:mp4?'video/mp4':'video/webm'}),extension:mp4?'mp4':'webm'}));
  send({type:'done',results,frames:frameCount,width,height,seconds:(performance.now()-started)/1000});
 } catch(error) {
  for(const output of outputs) {try{await output.cancel();}catch{}}
  send({type:'error',message:error instanceof Error?error.message:String(error)});
 } finally {input?.dispose();await encoder?.release();await renderer?.release();}
};

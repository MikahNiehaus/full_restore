from fastai.core import *
from fastai.vision import *
from matplotlib.axes import Axes
from .filters import IFilter, MasterFilter, ColorizerFilter
from .generators import gen_inference_deep, gen_inference_wide
from PIL import Image
import ffmpeg
import yt_dlp as youtube_dl
import gc
import requests
from io import BytesIO
import base64
from IPython import display as ipythondisplay
from IPython.display import HTML
from IPython.display import Image as ipythonimage
import cv2
import logging
import warnings
import os
import shutil
import numpy as np
import re
from pathlib import Path

# Suppress fastai UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="fastai")

# adapted from https://www.pyimagesearch.com/2016/04/25/watermarking-images-with-opencv-and-python/
def get_watermarked(pil_image: Image) -> Image:
    try:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        (h, w) = image.shape[:2]
        image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
        pct = 0.05
        full_watermark = cv2.imread(
            './resource_images/watermark.png', cv2.IMREAD_UNCHANGED
        )
        (fwH, fwW) = full_watermark.shape[:2]
        wH = int(pct * h)
        wW = int((pct * h / fwH) * fwW)
        watermark = cv2.resize(full_watermark, (wH, wW), interpolation=cv2.INTER_AREA)
        overlay = np.zeros((h, w, 4), dtype="uint8")
        (wH, wW) = watermark.shape[:2]
        overlay[h - wH - 10 : h - 10, 10 : 10 + wW] = watermark
        # blend the two images together using transparent overlays
        output = image.copy()
        cv2.addWeighted(overlay, 0.5, output, 1.0, 0, output)
        rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(rgb_image)
        return final_image
    except:
        # Don't want this to crash everything, so let's just not watermark the image for now.
        return pil_image


class ModelImageVisualizer:
    def __init__(self, filter: IFilter, results_dir: str = None):
        self.filter = filter
        self.results_dir = None if results_dir is None else Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _clean_mem(self):
        torch.cuda.empty_cache()
        # gc.collect()

    def _open_pil_image(self, path: Path) -> Image:
        return PIL.Image.open(path).convert('RGB')

    def _get_image_from_url(self, url: str) -> Image:
        response = requests.get(url, timeout=30, headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'})
        img = PIL.Image.open(BytesIO(response.content)).convert('RGB')
        return img

    def plot_transformed_image_from_url(
        self,
        url: str,
        path: str = 'test_images/image.png',
        results_dir:Path = None,
        figsize: Tuple[int, int] = (20, 20),
        render_factor: int = None,
        
        display_render_factor: bool = False,
        compare: bool = False,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        img = self._get_image_from_url(url)
        img.save(path)
        return self.plot_transformed_image(
            path=path,
            results_dir=results_dir,
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
            compare=compare,
            post_process = post_process,
            watermarked=watermarked,
        )

    def plot_transformed_image(
        self,
        path: str,
        results_dir:Path = None,
        figsize: Tuple[int, int] = (20, 20),
        render_factor: int = None,
        display_render_factor: bool = False,
        compare: bool = False,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        path = Path(path)
        if results_dir is None:
            results_dir = Path(self.results_dir)
        result = self.get_transformed_image(
            path, render_factor, post_process=post_process,watermarked=watermarked
        )
        orig = self._open_pil_image(path)
        if compare:
            self._plot_comparison(
                figsize, render_factor, display_render_factor, orig, result
            )
        else:
            self._plot_solo(figsize, render_factor, display_render_factor, result)

        orig.close()
        result_path = self._save_result_image(path, result, results_dir=results_dir)
        result.close()
        return result_path

    def _plot_comparison(
        self,
        figsize: Tuple[int, int],
        render_factor: int,
        display_render_factor: bool,
        orig: Image,
        result: Image,
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_image(
            orig,
            axes=axes[0],
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=False,
        )
        self._plot_image(
            result,
            axes=axes[1],
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _plot_solo(
        self,
        figsize: Tuple[int, int],
        render_factor: int,
        display_render_factor: bool,
        result: Image,
    ):
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        self._plot_image(
            result,
            axes=axes,
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _save_result_image(self, source_path: Path, image: Image, results_dir = None) -> Path:
        if results_dir is None:
            results_dir = Path(self.results_dir)
        result_path = results_dir / source_path.name
        image.save(result_path)
        return result_path

    def get_transformed_image(
        self, path: Path, render_factor: int = None, post_process: bool = True,
        watermarked: bool = True,
    ) -> Image:
        self._clean_mem()
        orig_image = self._open_pil_image(path)
        filtered_image = self.filter.filter(
            orig_image, orig_image, render_factor=render_factor,post_process=post_process
        )

        if watermarked:
            return get_watermarked(filtered_image)

        return filtered_image

    def _plot_image(
        self,
        image: Image,
        render_factor: int,
        axes: Axes = None,
        figsize=(20, 20),
        display_render_factor = False,
    ):
        if axes is None:
            _, axes = plt.subplots(figsize=figsize)
        axes.imshow(np.asarray(image) / 255)
        axes.axis('off')
        if render_factor is not None and display_render_factor:
            plt.text(
                10,
                10,
                'render_factor: ' + str(render_factor),
                color='white',
                backgroundcolor='black',
            )

    def _get_num_rows_columns(self, num_images: int, max_columns: int) -> Tuple[int, int]:
        columns = min(num_images, max_columns)
        rows = num_images // columns
        rows = rows if rows * columns == num_images else rows + 1
        return rows, columns


class VideoColorizer:
    def __init__(self, vis: ModelImageVisualizer):
        self.vis = vis
        workfolder = Path('./video')
        self.source_folder = workfolder / "source"
        self.bwframes_root = workfolder / "bwframes"
        self.audio_root = workfolder / "audio"
        self.colorframes_root = workfolder / "colorframes"
        self.result_folder = workfolder / "result"

    def _purge_images(self, dir):
        for f in os.listdir(dir):
            if re.search(r'.*?\.jpg', f):
                os.remove(os.path.join(dir, f))

    def _get_ffmpeg_probe(self, path:Path):
        try:
            probe = ffmpeg.probe(str(path))
            return probe
        except ffmpeg.Error as e:
            logging.error("ffmpeg error: {0}".format(e), exc_info=True)
            logging.error('stdout:' + e.stdout.decode('UTF-8'))
            logging.error('stderr:' + e.stderr.decode('UTF-8'))
            raise e
        except Exception as e:
            logging.error('Failed to instantiate ffmpeg.probe.  Details: {0}'.format(e), exc_info=True)   
            raise e

    def _get_fps(self, source_path: Path) -> str:
        probe = self._get_ffmpeg_probe(source_path)
        stream_data = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None,
        )
        return stream_data['avg_frame_rate']

    def _download_video_from_url(self, source_url, source_path: Path):
        if source_path.exists():
            source_path.unlink()

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': str(source_path),
            'retries': 30,
            'fragment-retries': 30
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source_url])

    def _extract_raw_frames(self, source_path: Path):
        bwframes_folder = self.bwframes_root / (source_path.stem)
        bwframe_path_template = str(bwframes_folder / '%5d.jpg')
        bwframes_folder.mkdir(parents=True, exist_ok=True)
        self._purge_images(bwframes_folder)

        process = (
            ffmpeg
                .input(str(source_path))
                .output(str(bwframe_path_template), format='image2', vcodec='mjpeg', **{'q:v':'0'})
                .global_args('-hide_banner')
                .global_args('-nostats')
                .global_args('-loglevel', 'error')
        )

        try:
            process.run()
        except ffmpeg.Error as e:
            logging.error("ffmpeg error: {0}".format(e), exc_info=True)
            logging.error('stdout:' + e.stdout.decode('UTF-8'))
            logging.error('stderr:' + e.stderr.decode('UTF-8'))
            raise e
        except Exception as e:
            logging.error('Errror while extracting raw frames from source video.  Details: {0}'.format(e), exc_info=True)   
            raise e    
    def _colorize_raw_frames(
        self, source_path: Path, render_factor: int = None, post_process: bool = True,
        watermarked: bool = True,
    ):
        colorframes_folder = self.colorframes_root / (source_path.stem)
        colorframes_folder.mkdir(parents=True, exist_ok=True)
        self._purge_images(colorframes_folder)
        bwframes_folder = self.bwframes_root / (source_path.stem)

        for img in progress_bar(os.listdir(str(bwframes_folder))):
            img_path = bwframes_folder / img

            if os.path.isfile(str(img_path)):
                color_image = self.vis.get_transformed_image(
                    str(img_path), render_factor=render_factor, post_process=post_process,watermarked=watermarked
                )
                color_image.save(str(colorframes_folder / img))
                
    def _build_video(self, source_path: Path) -> Path:
        colorized_path = self.result_folder / (
            source_path.name.replace('.mp4', '_no_audio.mp4')
        )
        colorframes_folder = self.colorframes_root / (source_path.stem)
        colorframes_path_template = str(colorframes_folder / '%5d.jpg')
        colorized_path.parent.mkdir(parents=True, exist_ok=True)
        if colorized_path.exists():
            colorized_path.unlink()
        fps = self._get_fps(source_path)

        process = (
            ffmpeg 
                .input(str(colorframes_path_template), format='image2', vcodec='mjpeg', framerate=fps) 
                .output(str(colorized_path), crf=17, vcodec='libx264')
                .global_args('-hide_banner')
                .global_args('-nostats')
                .global_args('-loglevel', 'error')
        )

        try:
            process.run()
        except ffmpeg.Error as e:
            logging.error("ffmpeg error: {0}".format(e), exc_info=True)
            logging.error('stdout:' + e.stdout.decode('UTF-8'))
            logging.error('stderr:' + e.stderr.decode('UTF-8'))
            raise e
        except Exception as e:
            logging.error('Errror while building output video.  Details: {0}'.format(e), exc_info=True)   
            raise e

        result_path = self.result_folder / source_path.name
        if result_path.exists():
            result_path.unlink()
        # Copy non-audio version in case adding back audio doesn't apply or fails.
        shutil.copyfile(str(colorized_path), str(result_path))        # Try to enhance the audio if it's a vintage film
        # Create temp directory for audio files if it doesn't exist
        audio_temp_dir = Path('audio_temp')
        audio_temp_dir.mkdir(exist_ok=True, parents=True)
        
        temp_audio_path = audio_temp_dir / 'temp_audio.wav'
        enhanced_audio_path = audio_temp_dir / 'temp_audio_enhanced.wav'
        audio_path_to_use = None        # Extract the audio from the original video with precise timing preservation
        # First get video frame rate for sync purposes
        video_info_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1 "{str(source_path)}"'
        try:
            video_info = subprocess.check_output(video_info_cmd, shell=True, text=True)
            frame_rate_str = video_info.strip().split('=')[1] if '=' in video_info else None
            logging.info(f"Video frame rate info: {frame_rate_str}")
        except Exception as e:
            logging.warning(f"Could not determine video frame rate: {e}")
        
        # Extract audio with precise timing preservation
        extract_cmd = (
            'ffmpeg -y -i "' + str(source_path) + '" -vn -acodec pcm_s16le -ar 48000 -ac 2 "'
            + str(temp_audio_path) + '" -hide_banner -nostats -loglevel error'
        )
        
        logging.info(f"Extracting audio with command: {extract_cmd}")
        # First check if audio extraction succeeds
        try:
            extract_result = os.system(extract_cmd)
        except Exception as e:
            logging.error(f"Error during audio extraction: {e}")
            extract_result = 1
        if extract_result == 0 and temp_audio_path.exists() and temp_audio_path.stat().st_size > 0:
            logging.info(f"Successfully extracted audio: {temp_audio_path} ({temp_audio_path.stat().st_size} bytes)")
            # Try to enhance the audio if possible - with multiple fallbacks
            try:
                # Try to import our vintage audio enhancer first
                sys.path.append(os.path.abspath('.'))
                try:
                    from vintage_audio_enhancer import VintageAudioEnhancer
                    audio_enhancer = VintageAudioEnhancer(verbose=True)  # Enable verbose mode for debugging
                    logging.info(f"Enhancing audio with vintage enhancer: {temp_audio_path} -> {enhanced_audio_path}")
                    enhance_success = audio_enhancer.enhance_vintage_audio(
                        str(temp_audio_path), str(enhanced_audio_path)
                    )
                    
                    if enhance_success and enhanced_audio_path.exists() and enhanced_audio_path.stat().st_size > 0:
                        audio_path_to_use = enhanced_audio_path
                        logging.info(f"Applied vintage audio enhancement: {enhanced_audio_path} ({enhanced_audio_path.stat().st_size} bytes)")
                    else:
                        # FIRST FALLBACK: Try regular audio enhancer
                        logging.info("Vintage enhancement failed, trying regular audio enhancer...")
                        try:
                            from audio_enhancer import AudioEnhancer
                            regular_enhancer = AudioEnhancer(verbose=True)
                            regular_enhance_path = audio_temp_dir / 'temp_audio_regular_enhanced.wav'
                            regular_success = regular_enhancer.enhance_audio(
                                str(temp_audio_path), str(regular_enhance_path)
                            )
                            
                            if regular_success and regular_enhance_path.exists() and regular_enhance_path.stat().st_size > 0:
                                audio_path_to_use = regular_enhance_path
                                logging.info(f"Applied regular audio enhancement: {regular_enhance_path}")
                            else:
                                # SECOND FALLBACK: Use original audio
                                audio_path_to_use = temp_audio_path
                                logging.info("Using original audio (all enhancements failed)")
                        except ImportError:
                            # SECOND FALLBACK: Use original audio
                            audio_path_to_use = temp_audio_path
                            logging.info("Regular audio enhancer not available, using original audio")
                except ImportError as ie:
                    # Try the regular audio enhancer as first fallback
                    logging.warning(f"Vintage audio enhancer import error: {ie}")
                    try:
                        from audio_enhancer import AudioEnhancer
                        regular_enhancer = AudioEnhancer(verbose=True)
                        regular_enhance_path = audio_temp_dir / 'temp_audio_regular_enhanced.wav'
                        regular_success = regular_enhancer.enhance_audio(
                            str(temp_audio_path), str(regular_enhance_path)
                        )
                        
                        if regular_success and regular_enhance_path.exists() and regular_enhance_path.stat().st_size > 0:
                            audio_path_to_use = regular_enhance_path
                            logging.info(f"Applied regular audio enhancement: {regular_enhance_path}")
                        else:
                            # Use original as last fallback
                            audio_path_to_use = temp_audio_path
                            logging.info("Using original audio (regular enhancement failed)")
                    except ImportError:
                        # Use original as last fallback
                        audio_path_to_use = temp_audio_path
                        logging.info("No audio enhancers available, using original audio")
            except Exception as e:
                logging.warning(f"Audio enhancement error: {e}")
                logging.warning(traceback.format_exc())
                # Final fallback: original audio
                audio_path_to_use = temp_audio_path
                logging.info("Using original audio due to enhancement errors")
          # Add audio to the colorized video with perfect sync
        if audio_path_to_use and audio_path_to_use.exists():
            # Use the enhanced audio if available with special flags for perfect sync
            mux_cmd = (
                'ffmpeg -y -i "' + str(colorized_path) + '" -i "' + str(audio_path_to_use) + '" '
                '-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k '
                '-vsync 0 -af "aresample=async=1" '
                '-metadata:s:a:0 "sync_audio=true" '
                '-shortest "' + str(result_path) + '" -hide_banner -nostats -loglevel error'
            )
            logging.info(f"Running audio mux command with enhanced audio: {mux_cmd}")
            os.system(mux_cmd)
            logging.info("Added enhanced audio to colorized video")
        else:
            # Fallback: Add original audio stream with sync preservation
            mux_cmd = (
                'ffmpeg -y -i "' + str(colorized_path) + '" -i "' + str(source_path) + '" '
                '-map 0:v:0 -map 1:a:0? -c:v copy -c:a aac -b:a 192k '
                '-vsync 0 -async 1 -shortest "' + str(result_path) + '" '
                '-hide_banner -nostats -loglevel error'
            )
            logging.info(f"Running audio mux command with original audio: {mux_cmd}")
            os.system(mux_cmd)
            logging.info("Added original audio stream to colorized video")
        
        # Clean up temporary audio files
        if temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
            except:
                pass
        if enhanced_audio_path.exists():
            try:
                enhanced_audio_path.unlink()
            except:
                pass
        
        logging.info('Video created here: ' + str(result_path))
        return result_path

    def colorize_from_url(
        self,
        source_url,
        file_name: str,
        render_factor: int = None,
        post_process: bool = True,
        watermarked: bool = True,

    ) -> Path:
        source_path = self.source_folder / file_name
        self._download_video_from_url(source_url, source_path)
        return self._colorize_from_path(
            source_path, render_factor=render_factor, post_process=post_process,watermarked=watermarked
        )

    def colorize_from_file_name(
        self, file_name: str, render_factor: int = None,  watermarked: bool = True, post_process: bool = True,
    ) -> Path:
        # Try to resolve the correct path for the input video file
        attempted_paths = []
        # 1. Absolute path or as given
        path1 = Path(file_name)
        if path1.is_absolute() and path1.exists():
            source_path = path1
        elif path1.exists():
            source_path = path1
        else:
            # 2. Try inputs/file_name
            path2 = Path('inputs') / file_name
            if path2.exists():
                source_path = path2
            else:
                # 3. Try self.source_folder/file_name
                path3 = self.source_folder / file_name
                if path3.exists():
                    source_path = path3
                else:
                    attempted_paths = [str(path1), str(path2), str(path3)]
                    raise Exception(
                        f"Video file could not be found. Tried the following paths: {attempted_paths}"
                    )
        return self._colorize_from_path(
            source_path, render_factor=render_factor,  post_process=post_process,watermarked=watermarked
        )

    def _colorize_from_path(
        self, source_path: Path, render_factor: int = None,  watermarked: bool = True, post_process: bool = True
    ) -> Path:
        if not source_path.exists():
            raise Exception(
                'Video at path specfied, ' + str(source_path) + ' could not be found.'
            )
        self._extract_raw_frames(source_path)
        self._colorize_raw_frames(
            source_path, render_factor=render_factor,post_process=post_process,watermarked=watermarked
        )
        return self._build_video(source_path)


def get_video_colorizer(render_factor: int = 21) -> VideoColorizer:
    return get_stable_video_colorizer(render_factor=render_factor)


def get_artistic_video_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeArtistic_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> VideoColorizer:
    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    # Move model to GPU if available
    if torch.cuda.is_available():
        learn.model = learn.model.to(torch.device('cuda'))
        logging.info("Using GPU for artistic video colorization")
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return VideoColorizer(vis)


def get_stable_video_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeVideo_gen',
    results_dir='result_images',
    render_factor: int = 21
) -> VideoColorizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    # Move model to GPU if available
    if torch.cuda.is_available():
        learn.model = learn.model.to(torch.device('cuda'))
        logging.info("Using GPU for stable video colorization")
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return VideoColorizer(vis)


def get_image_colorizer(
    root_folder: Path = Path('./'), render_factor: int = 35, artistic: bool = True
) -> ModelImageVisualizer:
    if artistic:
        return get_artistic_image_colorizer(root_folder=root_folder, render_factor=render_factor)
    else:
        return get_stable_image_colorizer(root_folder=root_folder, render_factor=render_factor)


def get_stable_image_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeStable_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> ModelImageVisualizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    # Move model to GPU if available
    if torch.cuda.is_available():
        learn.model = learn.model.to(torch.device('cuda'))
        logging.info("Using GPU for stable image colorization")
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def get_artistic_image_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeArtistic_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> ModelImageVisualizer:
    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    # Move model to GPU if available
    if torch.cuda.is_available():
        learn.model = learn.model.to(torch.device('cuda'))
        logging.info("Using GPU for artistic image colorization")
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def show_image_in_notebook(image_path: Path):
    ipythondisplay.display(ipythonimage(str(image_path)))


def show_video_in_notebook(video_path: Path):
    video = io.open(video_path, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(
        HTML(
            data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(
                encoded.decode('ascii')
            )
        )
    )

import os
import sys
import time
import argparse
import traceback
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our restoration and colorization function
from restore_and_colorize_simple import restore_and_colorize

class ImageHandler(FileSystemEventHandler):
    def __init__(self, input_dir, output_dir, processed_dir, scale=2):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processed_dir = processed_dir
        self.scale = scale
        self.processing = set()  # Track files being processed to avoid duplicates

    def on_created(self, event):
        if event.is_directory:
            return

        # Check if it's an image file
        if not event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')):
            return

        # Avoid processing the same file multiple times
        if event.src_path in self.processing:
            return

        self.processing.add(event.src_path)

        # Wait a moment to ensure the file is fully written
        time.sleep(1)

        try:
            print(f"\nNew file detected: {os.path.basename(event.src_path)}")
            print("Starting automatic restoration and colorization...")

            # Process the file
            restore_and_colorize(
                event.src_path,
                self.output_dir,
                self.processed_dir,
                self.scale
            )

        except Exception as e:
            print(f"Error processing new file: {e}")
            traceback.print_exc()
        finally:
            self.processing.remove(event.src_path)

def watch_directory(input_dir="inputs", output_dir="outputs", processed_dir="processed", scale=2):
    """
    Watch the input directory for new image files and process them automatically
    """
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Watching directory: {input_dir}")
    print(f"Any new images will be automatically restored and colorized")
    print(f"Processed images will be saved to: {output_dir}")
    print(f"Original files will be moved to: {processed_dir}")

    # Process any existing files in the directory
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')):
            file_path = os.path.join(input_dir, file)
            print(f"\nProcessing existing file: {file}")
            restore_and_colorize(file_path, output_dir, processed_dir, scale)

    # Set up the file system watcher
    event_handler = ImageHandler(input_dir, output_dir, processed_dir, scale)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)
    observer.start()

    print("\nWatcher started. Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a directory for new images and automatically restore and colorize them")
    parser.add_argument("-i", "--input", default="inputs", help="Input directory to watch (default: inputs)")
    parser.add_argument("-o", "--output", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("-p", "--processed", default="processed", help="Directory to move processed originals (default: processed)")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor for restoration (default: 2)")

    args = parser.parse_args()

    watch_directory(args.input, args.output, args.processed, args.scale)

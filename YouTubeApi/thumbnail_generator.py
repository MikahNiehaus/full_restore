#!/usr/bin/env python3
"""
Thumbnail Generator for YouTube Videos

This module generates thumbnails from videos for YouTube uploads.
"""

import os
import cv2
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

def extract_frame_from_video(video_path, frame_position=0.25, output_path=None):
    """
    Extract a frame from the video at the specified position.
    
    Args:
        video_path (str): Path to the video file
        frame_position (float): Position in the video to extract frame from (0.0-1.0)
        output_path (str): Path to save the extracted frame (optional)
        
    Returns:
        str: Path to the extracted frame
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        
        # Check if the video opened successfully
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return None
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame number to extract
        target_frame = int(total_frames * frame_position)
        
        # Set video position to the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Read the frame
        success, frame = cap.read()
        
        # Close the video file
        cap.release()
        
        if not success:
            print(f"[ERROR] Could not read frame {target_frame} from video: {video_path}")
            return None
            
        # Create output path if not provided
        if not output_path:
            temp_dir = tempfile.gettempdir()
            output_path = Path(temp_dir) / f"thumbnail_{Path(video_path).stem}.jpg"
        else:
            output_path = Path(output_path)
            
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for further processing
        pil_image = Image.fromarray(frame_rgb)
        
        # Ensure the image is in the correct orientation
        if hasattr(pil_image, '_getexif') and pil_image._getexif():
            orientation = 274  # EXIF orientation tag
            exif = pil_image._getexif()
            if exif and orientation in exif:
                # Handle image orientation
                if exif[orientation] == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    pil_image = pil_image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    pil_image = pil_image.rotate(90, expand=True)
        
        # Save the image
        pil_image.save(output_path, quality=95)
        print(f"[INFO] Extracted frame saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Error extracting frame from video: {e}")
        import traceback
        traceback.print_exc()
        return None

def enhance_thumbnail(image_path, output_path=None, text=None, logo_path=None):
    """
    Enhance a thumbnail image with text and effects for YouTube.
    
    Args:
        image_path (str): Path to the image file
        output_path (str): Path to save the enhanced thumbnail (optional)
        text (str): Text to overlay on the thumbnail (optional)
        logo_path (str): Path to a logo to add to the thumbnail (optional)
        
    Returns:
        str: Path to the enhanced thumbnail
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Resize to YouTube recommended size (1280x720)
        img = img.resize((1280, 720), Image.Resampling.LANCZOS)
        
        # Create output path if not provided
        if not output_path:
            output_path = str(Path(image_path).with_suffix('.thumb.jpg'))
        
        # Apply some enhancements
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)  # Increase contrast
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)  # Slightly increase brightness
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)  # Boost colors
        
        # Apply a slight blur to background for text visibility
        if text:
            # Create a semi-transparent overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Add gradient at the bottom for text visibility
            for i in range(200):
                alpha = int(180 * (i / 200))
                draw.line((0, img.height - i, img.width, img.height - i), 
                         fill=(0, 0, 0, alpha))
            
            # Composite the overlay with the image
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            
            # Add text
            draw = ImageDraw.Draw(img)
            
            # Use default font if custom font not available
            try:
                # Try to use a bold font if available
                font_path = "arial.ttf"  # Adjust to a font available on your system
                font_size = 60
                font = ImageFont.truetype(font_path, font_size)
            except:
                # Fall back to default font
                font = ImageFont.load_default()
                font_size = 30
            
            # Measure text size
            lines = text.split('\n')
            line_height = font_size + 10
            
            # Position text at bottom
            text_y = img.height - line_height * len(lines) - 20
            
            # Add text with shadow effect
            for i, line in enumerate(lines):
                y_position = text_y + i * line_height
                # Draw shadow (offset)
                draw.text((22, y_position + 2), line, font=font, fill=(0, 0, 0))
                # Draw actual text
                draw.text((20, y_position), line, font=font, fill=(255, 255, 255))
        
        # Add logo if provided
        if logo_path and Path(logo_path).exists():
            try:
                logo = Image.open(logo_path)
                # Resize logo to reasonable size (e.g. 100x100 or maintaining aspect ratio)
                logo_size = (100, 100)
                logo = logo.resize(logo_size, Image.Resampling.LANCZOS)
                
                # Position logo in top right corner with padding
                logo_position = (img.width - logo_size[0] - 20, 20)
                
                # If logo has transparency (RGBA), use it as is
                if logo.mode == 'RGBA':
                    img.paste(logo, logo_position, logo)
                else:
                    img.paste(logo, logo_position)
            except Exception as e:
                print(f"[WARNING] Error adding logo to thumbnail: {e}")
        
        # Save the enhanced thumbnail
        img.save(output_path, quality=95)
        print(f"[INFO] Enhanced thumbnail saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Error enhancing thumbnail: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_thumbnail(video_path, output_path=None, text=None, logo_path=None, frame_position=0.25):
    """
    Generate a thumbnail from a video file.
    
    Args:
        video_path (str): Path to the video file
        output_path (str): Path to save the generated thumbnail (optional)
        text (str): Text to overlay on the thumbnail (optional)
        logo_path (str): Path to a logo to add to the thumbnail (optional)
        frame_position (float): Position in the video to extract frame from (0.0-1.0)
        
    Returns:
        str: Path to the generated thumbnail
    """
    # Extract frame from video
    frame_path = extract_frame_from_video(video_path, frame_position)
    
    if not frame_path:
        return None
    
    # Enhance the thumbnail
    enhanced_path = enhance_thumbnail(frame_path, output_path, text, logo_path)
    
    # Clean up temporary frame if we created one
    if frame_path != output_path and os.path.exists(frame_path):
        try:
            os.remove(frame_path)
        except:
            pass
    
    return enhanced_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a thumbnail from a video.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--output", "-o", help="Path to save the thumbnail")
    parser.add_argument("--text", "-t", help="Text to overlay on the thumbnail")
    parser.add_argument("--logo", "-l", help="Path to a logo to add to the thumbnail")
    parser.add_argument("--position", "-p", type=float, default=0.25,
                       help="Position in the video to extract frame from (0.0-1.0)")
    
    args = parser.parse_args()
    
    thumbnail_path = generate_thumbnail(
        args.video, args.output, args.text, args.logo, args.position
    )
    
    if thumbnail_path:
        print(f"Thumbnail generated successfully: {thumbnail_path}")
    else:
        print("Failed to generate thumbnail")

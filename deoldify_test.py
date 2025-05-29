import os
import sys

# Add DeOldify to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeOldify'))

def check_deoldify_paths():
    """Check DeOldify paths and model presence"""
    print("Checking DeOldify setup...")
    
    # Check if DeOldify directory exists
    deoldify_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeOldify')
    models_path = os.path.join(deoldify_path, 'models')
    
    print(f"DeOldify path: {deoldify_path}")
    print(f"Models path: {models_path}")
    print(f"DeOldify directory exists: {os.path.exists(deoldify_path)}")
    print(f"Models directory exists: {os.path.exists(models_path)}")
    
    # Check model files
    stable_model = os.path.join(models_path, 'ColorizeStable_gen.pth')
    artistic_model = os.path.join(models_path, 'ColorizeArtistic_gen.pth')
    
    print(f"ColorizeStable_gen.pth exists: {os.path.exists(stable_model)}")
    print(f"ColorizeArtistic_gen.pth exists: {os.path.exists(artistic_model)}")
    
    # Check if we can import DeOldify modules
    try:
        from deoldify.visualize import get_image_colorizer
        print("Successfully imported get_image_colorizer")
        
        # Try initializing the colorizer with both models
        try:
            print("\nTrying to initialize stable colorizer...")
            stable_colorizer = get_image_colorizer(artistic=False)
            print("Stable colorizer initialized successfully!")
        except Exception as e:
            print(f"Error initializing stable colorizer: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            print("\nTrying to initialize artistic colorizer...")
            artistic_colorizer = get_image_colorizer(artistic=True)
            print("Artistic colorizer initialized successfully!")
        except Exception as e:
            print(f"Error initializing artistic colorizer: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"Error importing DeOldify modules: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    check_deoldify_paths()

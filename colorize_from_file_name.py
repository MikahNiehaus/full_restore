def colorize_from_file_name(
        self, file_name: str, render_factor: int = None, watermarked: bool = True, post_process: bool = True,
    ) -> Path:
        # Try multiple locations in order of priority
        file_path = Path(file_name)
        paths_to_try = [
            file_path,  # 1. Use as-is (full path or relative)
            Path('inputs') / file_path.name,  # 2. Try in inputs/
            self.source_folder / file_path.name,  # 3. Try in video/source/
        ]
        
        # Handle the edge case where the input might be 'inputs/file' and it doesn't exist
        # because it's being checked as 'inputs/inputs/file'
        if str(file_path).startswith('inputs'):
            clean_name = str(file_path).replace('inputs/', '').replace('inputs\\', '')
            paths_to_try.insert(1, Path('inputs') / clean_name)
            
        # Try each path
        found_path = None
        for path in paths_to_try:
            if path.exists():
                found_path = path
                break
                
        if found_path is None:
            print(f"[DEBUG] File not found. Tried paths:")
            for i, path in enumerate(paths_to_try):
                print(f"  {i+1}. {path}")
            print(f"[DEBUG] Current working directory: {os.getcwd()}")
            raise Exception(f'Video not found. Tried multiple paths including inputs/ and {self.source_folder}/')
        return self._colorize_from_path(
            found_path, render_factor=render_factor, post_process=post_process, watermarked=watermarked
        )

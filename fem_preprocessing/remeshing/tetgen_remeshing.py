import subprocess
import os
import shutil


def run_tetgen(input_file, tetgen_path, base_output_folder, element_size=0.025):
    output_folder = base_output_folder

    os.makedirs(output_folder, exist_ok=True)

    # TetGen switches:
    # -p: tetrahedralize the input surface geometry
    # -q: enable quality mesh generation
    # -g: generate visualization output
    # -a: impose the specified maximum tetrahedral volume
    command = [
        tetgen_path,
        f"-pqga{element_size}",
        input_file
    ]

    try:
        subprocess.run(
            command,
            check=True,
            cwd=os.path.dirname(input_file)
        )

        print("TetGen ran successfully!")

        prefix = os.path.splitext(os.path.basename(input_file))[0]
        working_dir = os.path.dirname(input_file)

        output_files = [
            f for f in os.listdir(working_dir)
            if f.startswith(prefix) and not f.endswith(".stl")
        ]

        if not output_files:
            print("No TetGen output files found!")
        else:
            print("Generated files:", output_files)

            for file in output_files:
                source_file = os.path.join(working_dir, file)

                name_parts = file.split(prefix, 1)
                new_filename = f"{prefix}_size{element_size}{name_parts[1]}"

                destination_file = os.path.join(
                    output_folder,
                    new_filename
                )

                shutil.move(source_file, destination_file)

                print(
                    f"Moved and renamed {file} -> {new_filename}"
                )

    except subprocess.CalledProcessError as e:
        print(f"Error running TetGen: {e}")


if __name__ == "__main__":
    input_stl = r"path\to\input_meniscus.stl"
    tetgen_exe = r"path\to\tetgen.exe"
    base_output_dir = r"path\to\output_directory"

    element_size = #insert desired element size

    run_tetgen(
        input_stl,
        tetgen_exe,
        base_output_dir,
        element_size
    )



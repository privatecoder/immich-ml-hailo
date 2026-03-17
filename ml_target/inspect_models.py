import glob
import os
import hailo_platform as hpf


def main() -> None:
    models_dir = "/app/models"
    paths = sorted(glob.glob(os.path.join(models_dir, "*.hef")))
    if not paths:
        print("No .hef models found in", models_dir)
        return

    for p in paths:
        print("=" * 80)
        print("MODEL:", p)
        hef = hpf.HEF(p)

        print("  INPUTS:")
        for i in hef.get_input_vstream_infos():
            f = i.format
            print("   -", i.name, "shape=", i.shape, "type=", f.type, "order=", f.order)

        print("  OUTPUTS:")
        for o in hef.get_output_vstream_infos():
            f = o.format
            print("   -", o.name, "shape=", o.shape, "type=", f.type, "order=", f.order)


if __name__ == "__main__":
    main()

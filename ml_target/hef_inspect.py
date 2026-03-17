import hailo_platform as hpf


def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 -m ml_target.hef_inspect /path/to/model.hef")
        raise SystemExit(2)

    hef_path = sys.argv[1]
    hef = hpf.HEF(hef_path)

    print("HEF:", hef_path)
    print("\n=== INPUTS ===")
    for i in hef.get_input_vstream_infos():
        f = i.format
        print(" ", i.name, "shape=", i.shape, "type=", f.type, "order=", f.order, "flags=", f.flags)

    print("\n=== OUTPUTS ===")
    for o in hef.get_output_vstream_infos():
        f = o.format
        print(" ", o.name, "shape=", o.shape, "type=", f.type, "order=", f.order, "flags=", f.flags)


if __name__ == "__main__":
    main()

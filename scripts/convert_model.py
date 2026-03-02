#!/usr/bin/env python3
"""
Extract BRISQUE model data from OpenCV YAML files and output as Go source code.

Usage:
    # First, download the model files:
    curl -L -o brisque_model_live.yml \
      https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml
    curl -L -o brisque_range_live.yml \
      https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml

    # Then run this script:
    python3 convert_model.py brisque_model_live.yml brisque_range_live.yml > ../model_default.go

    # Or, if you have opencv-contrib-python installed, use --opencv mode
    # which loads via cv2.ml.SVM and verifies the data:
    python3 convert_model.py --opencv brisque_model_live.yml brisque_range_live.yml > ../model_default.go
"""

import sys
import re
import argparse


def parse_opencv_yaml_manual(model_path, range_path):
    """Parse OpenCV YAML files without requiring cv2."""

    # --- Parse model file ---
    with open(model_path, "r") as f:
        model_text = f.read()

    # Extract gamma
    gamma_match = re.search(r"gamma:\s*([\d.eE+\-]+)", model_text)
    if not gamma_match:
        raise ValueError("Could not find gamma in model file")
    gamma = float(gamma_match.group(1))

    # Extract sv_total
    sv_total_match = re.search(r"sv_total:\s*(\d+)", model_text)
    if not sv_total_match:
        raise ValueError("Could not find sv_total in model file")
    sv_total = int(sv_total_match.group(1))

    # Extract support vectors
    # Find the support_vectors section
    sv_section = re.search(
        r"support_vectors:\s*\n(.*?)(?=\n\s*decision_functions:)",
        model_text,
        re.DOTALL,
    )
    if not sv_section:
        raise ValueError("Could not find support_vectors section")

    # Each SV is a multi-line "- [ ... ]" block. Collect bracket-delimited entries.
    sv_text = sv_section.group(1)
    support_vectors = []
    current = []
    in_bracket = False
    for line in sv_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- ["):
            in_bracket = True
            # Extract content after "- ["
            content = stripped[3:]
            if content.endswith("]"):
                content = content[:-1]
                in_bracket = False
            nums = [float(x.strip()) for x in content.split(",") if x.strip()]
            current.extend(nums)
            if not in_bracket:
                support_vectors.append(current)
                current = []
        elif in_bracket:
            content = stripped
            if content.endswith("]"):
                content = content[:-1]
                in_bracket = False
            nums = [float(x.strip()) for x in content.split(",") if x.strip()]
            current.extend(nums)
            if not in_bracket:
                support_vectors.append(current)
                current = []

    if len(support_vectors) != sv_total:
        raise ValueError(
            f"Expected {sv_total} support vectors, got {len(support_vectors)}"
        )
    for i, sv in enumerate(support_vectors):
        if len(sv) != 36:
            raise ValueError(f"SV {i}: expected 36 features, got {len(sv)}")

    # Extract decision functions (rho and alpha)
    df_section = re.search(
        r"decision_functions:\s*\n(.*?)$", model_text, re.DOTALL
    )
    if not df_section:
        raise ValueError("Could not find decision_functions section")

    df_text = df_section.group(1)

    rho_match = re.search(r"rho:\s*([\d.eE+\-]+)", df_text)
    if not rho_match:
        raise ValueError("Could not find rho in decision_functions")
    rho = float(rho_match.group(1))

    # Extract alpha values
    alpha_match = re.search(r"alpha:\s*\[\s*(.*?)\s*\]", df_text, re.DOTALL)
    if not alpha_match:
        raise ValueError("Could not find alpha in decision_functions")
    alpha_text = alpha_match.group(1)
    alphas = [float(x.strip()) for x in alpha_text.split(",") if x.strip()]

    if len(alphas) != sv_total:
        raise ValueError(f"Expected {sv_total} alphas, got {len(alphas)}")

    # --- Parse range file ---
    with open(range_path, "r") as f:
        range_text = f.read()

    # The range file contains a single matrix under key "range"
    # Format: !opencv-matrix with rows, cols, dt, data
    range_data_match = re.search(
        r"range:.*?data:\s*\[\s*(.*?)\s*\]", range_text, re.DOTALL
    )
    if not range_data_match:
        raise ValueError("Could not find range data matrix")

    range_values = [
        float(x.strip())
        for x in range_data_match.group(1).split(",")
        if x.strip()
    ]

    # Determine format: could be 2x36 (lower row, upper row) or 36x2 (lower,upper per feature)
    rows_match = re.search(r"range:.*?rows:\s*(\d+)", range_text, re.DOTALL)
    cols_match = re.search(r"range:.*?cols:\s*(\d+)", range_text, re.DOTALL)
    rows = int(rows_match.group(1)) if rows_match else None
    cols = int(cols_match.group(1)) if cols_match else None

    print(f"// Range matrix: {rows} rows x {cols} cols", file=sys.stderr)
    print(f"// Total range values: {len(range_values)}", file=sys.stderr)

    mins = []
    maxs = []
    if rows == 2 and cols == 36:
        # Row 0 = lower bounds, Row 1 = upper bounds
        mins = range_values[:36]
        maxs = range_values[36:]
    elif rows == 36 and cols == 2:
        # Each row = [lower, upper]
        for i in range(36):
            mins.append(range_values[i * 2])
            maxs.append(range_values[i * 2 + 1])
    else:
        raise ValueError(
            f"Unexpected range matrix dimensions: {rows}x{cols}"
        )

    return gamma, rho, sv_total, alphas, support_vectors, mins, maxs


def parse_opencv_yaml_cv2(model_path, range_path):
    """Parse using cv2 for verification."""
    import cv2
    import numpy as np

    # Load SVM model
    svm = cv2.ml.SVM_load(model_path)
    gamma = svm.getGamma()
    sv = svm.getSupportVectors()  # numpy array: (sv_total, 36)
    sv_total = sv.shape[0]

    # Get decision function data
    # OpenCV stores alpha and rho in the decision function
    # For EPS_SVR there's one decision function
    # We need to use the internal representation

    # Alternative: parse YAML manually for rho and alpha
    # since cv2 doesn't expose these directly in Python easily
    import cv2

    fs = cv2.FileStorage(model_path, cv2.FILE_STORAGE_READ)
    root = fs.getNode("opencv_ml_svm")
    df_node = root.getNode("decision_functions")
    df0 = df_node.getNode(0)
    rho = df0.getNode("rho").real()

    # Alpha is stored as raw data, need to parse from YAML manually
    # Fall back to manual parsing for alpha
    with open(model_path, "r") as f:
        model_text = f.read()
    df_section = re.search(
        r"decision_functions:\s*\n(.*?)$", model_text, re.DOTALL
    )
    alpha_match = re.search(
        r"alpha:\s*\[\s*(.*?)\s*\]", df_section.group(1), re.DOTALL
    )
    alphas = [
        float(x.strip()) for x in alpha_match.group(1).split(",") if x.strip()
    ]

    # Support vectors as list of lists
    support_vectors = sv.tolist()

    # Load range data
    fs_range = cv2.FileStorage(range_path, cv2.FILE_STORAGE_READ)
    range_mat = fs_range.getNode("range").mat()
    print(f"// Range matrix shape: {range_mat.shape}", file=sys.stderr)

    rows, cols = range_mat.shape
    mins = []
    maxs = []
    if rows == 2 and cols == 36:
        mins = range_mat[0].tolist()
        maxs = range_mat[1].tolist()
    elif rows == 36 and cols == 2:
        for i in range(36):
            mins.append(float(range_mat[i, 0]))
            maxs.append(float(range_mat[i, 1]))
    else:
        raise ValueError(f"Unexpected range shape: {range_mat.shape}")

    fs.release()
    fs_range.release()

    return gamma, rho, sv_total, alphas, support_vectors, mins, maxs


def format_float(v):
    """Format a float64 for Go source code."""
    if v == 0.0:
        return "0"
    # Use enough precision to round-trip float64
    s = f"{v:.17e}"
    # Clean up trailing zeros in mantissa
    if "e" in s:
        mantissa, exp = s.split("e")
        # Remove trailing zeros after decimal point
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        s = f"{mantissa}e{exp}"
    return s


def generate_go(gamma, rho, sv_total, alphas, support_vectors, mins, maxs):
    """Generate Go source code with embedded model data."""
    lines = []
    lines.append("package brisque")
    lines.append("")
    lines.append('import "github.com/matej/brisque-go/internal/svr"')
    lines.append("")
    lines.append(
        "// This file is auto-generated by scripts/convert_model.py"
    )
    lines.append(
        "// from the OpenCV BRISQUE default model files:"
    )
    lines.append(
        "//   brisque_model_live.yml (SVM model trained on LIVE-R2 database)"
    )
    lines.append(
        "//   brisque_range_live.yml (feature scaling ranges)"
    )
    lines.append(
        "// Source: https://github.com/opencv/opencv_contrib/tree/master/modules/quality/samples"
    )
    lines.append("")

    # Scale mins
    lines.append("// defaultScaleMins are the per-feature lower bounds for [-1,1] scaling.")
    lines.append("var defaultScaleMins = [36]float64{")
    for i, v in enumerate(mins):
        comma = "," if i < 35 else ","
        lines.append(f"\t{format_float(v)}{comma} // feature {i}")
    lines.append("}")
    lines.append("")

    # Scale maxs
    lines.append("// defaultScaleMaxs are the per-feature upper bounds for [-1,1] scaling.")
    lines.append("var defaultScaleMaxs = [36]float64{")
    for i, v in enumerate(maxs):
        comma = "," if i < 35 else ","
        lines.append(f"\t{format_float(v)}{comma} // feature {i}")
    lines.append("}")
    lines.append("")

    # SVR model constructor
    lines.append(f"// defaultSVRModel returns the pre-trained SVR model with {sv_total} support vectors.")
    lines.append("func defaultSVRModel() *svr.Model {")
    lines.append("\treturn &svr.Model{")
    lines.append(f"\t\tGamma: {format_float(gamma)},")
    lines.append(f"\t\tRho:   {format_float(rho)},")
    lines.append(f"\t\tNSV:   {sv_total},")
    lines.append(f"\t\tAlpha: defaultAlpha,")
    lines.append(f"\t\tSupportVectors: defaultSupportVectors,")
    lines.append("\t}")
    lines.append("}")
    lines.append("")

    # Alpha values
    lines.append(f"// defaultAlpha contains {sv_total} alpha (dual) coefficients.")
    lines.append(f"var defaultAlpha = []float64{{")
    for i, a in enumerate(alphas):
        lines.append(f"\t{format_float(a)},")
    lines.append("}")
    lines.append("")

    # Support vectors (flat N*36 layout)
    lines.append(
        f"// defaultSupportVectors contains {sv_total} support vectors in flat [N*36]float64 layout."
    )
    lines.append(f"var defaultSupportVectors = []float64{{")
    for i, sv in enumerate(support_vectors):
        lines.append(f"\t// SV {i}")
        row = ", ".join(format_float(v) for v in sv)
        lines.append(f"\t{row},")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenCV BRISQUE model files to Go source code"
    )
    parser.add_argument("model", help="Path to brisque_model_live.yml")
    parser.add_argument("range", help="Path to brisque_range_live.yml")
    parser.add_argument(
        "--opencv",
        action="store_true",
        help="Use cv2 for parsing (requires opencv-contrib-python)",
    )
    args = parser.parse_args()

    if args.opencv:
        gamma, rho, sv_total, alphas, svecs, mins, maxs = (
            parse_opencv_yaml_cv2(args.model, args.range)
        )
    else:
        gamma, rho, sv_total, alphas, svecs, mins, maxs = (
            parse_opencv_yaml_manual(args.model, args.range)
        )

    print(f"// Parsed: gamma={gamma}, rho={rho}, sv_total={sv_total}", file=sys.stderr)
    print(f"// Alpha count: {len(alphas)}", file=sys.stderr)
    print(f"// SV count: {len(svecs)}, features per SV: {len(svecs[0])}", file=sys.stderr)
    print(f"// Scale mins count: {len(mins)}", file=sys.stderr)
    print(f"// Scale maxs count: {len(maxs)}", file=sys.stderr)

    go_code = generate_go(gamma, rho, sv_total, alphas, svecs, mins, maxs)
    print(go_code)


if __name__ == "__main__":
    main()

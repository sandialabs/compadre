#!/usr/bin/env python3
"""
Create a snapshot from the Compadre repository containing this script into the
Trilinos repository pointed to, omitting the kokkos, kokkos-kernels, python,
and scripts directories.

Note:  This will assert that both repositories are in a clean state.

Warning:  This will clean out any locally-ignored files in Compadre (e.g.,
    ignored via .git/info/exclude) to avoid copying them over and then
    committing them to Trilinos.  Be sure you don't have any ignored files in
    Compadre that you want to keep before running this script.

To test this script to ensure that it's working correctly, simply use

    pytest snapshot_into_trilinos.py
"""
import os
import pytest
import sys




def parse_arguments(argv):
    """
    Parse the command line arguments to the script.

    Parameters:
        argv (list):  The command line arguments to be parsed.

    Returns:
        dict:  A mapping from the options to their values.
    """
    import argparse, textwrap
    width = 79
    description = __doc__
    description = ("[ Snapshot Compadre Into Trilinos ]".center(width, "-") +
                   "\n" + description)
    examples = """
        Show what's going to happen without actually doing the snapshot::

            ./snapshot_into_trilinos.py \\
                --trilinos-dir /path/to/Trilinos \\
                --dry-run

        Actually do the snapshot::

            ./snapshot_into_trilinos.py \\
                --trilinos-dir /path/to/Trilinos
    """
    examples = textwrap.dedent(examples)
    examples = "[ Examples ]".center(width, "-") + "\n\n" + examples
    parser = argparse.ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-t", "--trilinos-dir", dest="trilinos_dir",
                        action="store", required=True, default=None,
                        help="The path (relative or absolute) to the root of "
                        "the Trilinos repository you wish to snapshot "
                        "Compadre into.")
    parser.add_argument("-d", "--dry-run", dest="dry_run", action="store_true",
                        default=False, help="Show what will happen, but don't "
                        "actually do it.")
    return parser.parse_args(argv)



def test_parse_arguments():
    """
    Test that the :func:`parse_arguments` function works as intended.
    """
    options = parse_arguments("--trilinos-dir /path/to/Trilinos --dry-run".
                              split())
    assert (options.trilinos_dir == "/path/to/Trilinos" and
            options.dry_run == True)
    options = parse_arguments("-t /path/to/Trilinos -d".split())
    assert (options.trilinos_dir == "/path/to/Trilinos" and
            options.dry_run == True)
    options = parse_arguments("-t /some/other/dir".split())
    assert (options.trilinos_dir == "/some/other/dir" and
            options.dry_run == False)



def check_trilinos_dir_is_root(trilinos_dir):
    """
    Check to ensure the Trilinos directory specified is indeed the root of the
    Trilinos repository by ensuring it contains a ``ProjectName.cmake`` that
    specifies Trilinos as the project name.
    """
    import re
    try:
        with open(os.path.join(trilinos_dir, "ProjectName.cmake"), "r") as f:
            data = f.read().lower()
            if not re.search(r"set\s*\(\s*project_name\s+trilinos\s*\)", data):
                sys.exit(f"Error:  {trilinos_dir}/ProjectName.cmake does not "
                         "set the project name to Trilinos.  Make sure you "
                         "point to the root of the Trilinos repository.")
    except FileNotFoundError:
        sys.exit(f"Error:  {trilinos_dir} does not contain a "
                 "ProjectName.cmake.  Make sure you point to the root of the "
                 "Trilinos repository.")



def test_check_trilinos_dir_is_root(tmpdir):
    """
    Test that the :func:`check_trilinos_dir_is_root` function works as
    intended.
    """
    # Ensure the check fails if trilinos_dir doesn't contain ProjectName.cmake.
    trilinos_dir = tmpdir.mkdir("trilinos_dir")
    with pytest.raises(SystemExit):
        check_trilinos_dir_is_root(trilinos_dir)

    # Ensure the check fails if ProjectName.cmake doesn't set the project name
    # to Trilinos.
    file_to_check = trilinos_dir.join("ProjectName.cmake")
    file_to_check.write("Doesn't set PROJECT_NAME to Trilinos")
    with pytest.raises(SystemExit):
        check_trilinos_dir_is_root(trilinos_dir)

    # Ensure the check passes if ProjectName.cmake does set the project name to
    # Trilinos.
    file_to_check.write("SET(PROJECT_NAME Trilinos)")
    check_trilinos_dir_is_root(trilinos_dir)



def create_directory_variables(trilinos_dir, verbose=False):
    """
    Use the given path to Trilinos to create a handful of other variables
    pointing to:
    *  The python_utils directory, so we can import the SnapshotDir utility.
    *  The Compadre repository containing this script.
    *  The location to where Compadre will be shapshotted.

    Parameters:
        trilinos_dir (str):  The path to Trilinos given on the command line.
        verbose (bool):  Whether or not to print out details.

    Returns:
        tuple:  The original Compadre directory, and the location to be
            snapshotted to.
    """
    trilinos_dir = os.path.abspath(trilinos_dir)
    python_utils_dir = os.path.join(trilinos_dir, "cmake/tribits/python_utils")
    sys.path.append(python_utils_dir)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    compadre_orig_dir = os.path.abspath(os.path.join(script_dir, ".."))
    compadre_dest_dir = os.path.join(trilinos_dir, "packages/compadre")
    if verbose:
        print(f"Trilinos repository root:  {trilinos_dir}\n"
              f"Using snapshot-dir.py from:  {python_utils_dir}\n"
              f"Snapshotting Compadre from:  {compadre_orig_dir}\n"
              f"                      into:  {compadre_dest_dir}")
    return (compadre_orig_dir, compadre_dest_dir)



def test_create_directory_variables(capfd):
    """
    Test that the :func:`create_directory_variables` function works as
    intended.
    """
    trilinos_dir = "/path/to/Trilinos"
    compadre_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), ".."))
    (orig_dir, dest_dir) = create_directory_variables(trilinos_dir)
    assert (orig_dir == compadre_dir and
            dest_dir == "/path/to/Trilinos/packages/compadre" and
            os.path.join(trilinos_dir, "cmake/tribits/python_utils") in
                sys.path)
    out, err = capfd.readouterr()
    assert out == ""
    create_directory_variables(trilinos_dir, verbose=True)
    out, err = capfd.readouterr()
    expected = (f"Trilinos repository root:  {trilinos_dir}\n"
                f"Using snapshot-dir.py from:  {trilinos_dir}/cmake/tribits/"
                    "python_utils\n"
                f"Snapshotting Compadre from:  {compadre_dir}\n"
                f"                      into:  {trilinos_dir}/packages/"
                    "compadre\n")
    assert out == expected



def create_snapshot_dir_args(orig_dir, dest_dir, dry_run=False):
    """
    Create the arguments to pass to the SnapshotDir utility from TriBITS.

    Parameters:
        orig_dir (str):  The path to the Compadre repository to be snapshotted.
        dest_dir (str):  The path to where Compadre will be snapshotted.
        dry_run (bool):  Whether or not to show what will happen without
            actually doing it.

    Returns:
        list:  The arguments to be passed to the utility.
    """
    args = (f"--orig-dir {orig_dir}/ "
            f"--dest-dir {dest_dir}/ "
            "--exclude kokkos kokkos-kernels python scripts "
            "cmake/detect_trilinos_opts.cmake "
            "examples/Python_3D_Convergence.py.in "
            "--clean-ignored-files-orig-dir")
    if dry_run:
        args += " --show-defaults"
    return args.split(" ")



def test_create_snapshot_dir_args():
    """
    Test that the :func:`create_snapshot_dir_args` function works as intended.
    """
    orig = "from_here"
    dest = "to_there"
    args = create_snapshot_dir_args(orig, dest)
    expected = (f"--orig-dir {orig}/ --dest-dir {dest}/ --exclude kokkos "
                "kokkos-kernels python scripts cmake/bob.cmake "
                "cmake/detect_trilinos_opts.cmake "
                "examples/Python_3D_Convergence.py.in "
                "--clean-ignored-files-orig-dir")
    assert args == expected.split()
    args = create_snapshot_dir_args(orig, dest, dry_run=True)
    expected += " --show-defaults"
    assert args == expected.split()



def snapshot(snapshot_dir_args):
    """
    Perform the snapshot using the SnapshotDir utility from TriBITS.

    Parameters:
        shapshot_dir_args (str):  The arguments to pass to SnapshotDir.

    Returns:
        bool:  Whether or not the snapshot was successful.

    Note:
        The SnapshotDir utility is already unit tested within TriBITS, so
        there's no need for an additional test here.
    """
    import SnapshotDir
    return SnapshotDir.snapshotDirMainDriver(snapshot_dir_args)



if __name__ == "__main__":
    options = parse_arguments(sys.argv[1:])
    check_trilinos_dir_is_root(options.trilinos_dir)
    (orig_dir, dest_dir) = create_directory_variables(options.trilinos_dir,
                                                      options.dry_run)
    success = snapshot(create_snapshot_dir_args(orig_dir, dest_dir,
                                                options.dry_run))
    return_code = 0 if success else 1
    sys.exit(return_code)

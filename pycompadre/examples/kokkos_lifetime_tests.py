"""KokkosParser: Kokkos runtime lifetime, and the arguments Kokkos is initialized with.

Kokkos aborts if a View is deallocated after Kokkos::finalize. pycompadre objects (GMLS, ParticleHelper, ...)
own Views, so the Kokkos runtime must stay alive until the last of them is gone, whatever order Python
releases objects in. Each scenario runs in its own interpreter, because the failures happen during object
release and interpreter teardown, and a Kokkos abort would otherwise take down the test runner.

KokkosParser takes the Kokkos arguments from sys.orig_argv (see the KokkosParser documentation); the scenarios
in TestKokkosParserArguments pass command-line arguments to their interpreters to check that.

Registered as its own ctest (PyCOMPADRE_Kokkos_Lifetime) rather than discovered with the other tests.
"""

import os
import subprocess
import sys
import textwrap
import unittest

import pycompadre

# code shared by every scenario: builds a GMLS problem and generates alphas
SETUP = textwrap.dedent("""
    import sys
    import numpy as np
    import pycompadre

    def make_gmls():
        rng = np.random.default_rng(0)
        source_sites = rng.random((200, 2))
        target_sites = rng.random((10, 2))
        gmls = pycompadre.GMLS(2, 2)
        helper = pycompadre.ParticleHelper(gmls)
        helper.generateKDTree(source_sites)
        helper.generateNeighborListsFromKNNSearchAndSet(target_sites, 2, 2, 1.5)
        gmls.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
        gmls.generateAlphas()
        return gmls, helper
""")

KOKKOS_ABORT_MESSAGE = "after Kokkos::finalize"


def run_scenario(body, command_line_args=()):
    """Runs SETUP followed by body in a fresh interpreter, given command_line_args; returns the CompletedProcess."""
    env = dict(os.environ)
    # make the pycompadre imported here importable in the child
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(pycompadre.__file__)))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_parent, env.get("PYTHONPATH")]))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_PROC_BIND", "false")
    return subprocess.run([sys.executable, "-c", SETUP + textwrap.dedent(body), *command_line_args],
                          env=env, capture_output=True, text=True, timeout=120)


class ScenarioTestCase(unittest.TestCase):

    def assertScenario(self, result, expected_returncode=0, expected_stdout=None, expected_stderr=None):
        details = "\nreturn code: %d\n--- stdout ---\n%s\n--- stderr (last 2000 chars) ---\n%s" % (
            result.returncode, result.stdout, result.stderr[-2000:])
        self.assertNotIn(KOKKOS_ABORT_MESSAGE, result.stderr,
                         "a Kokkos View was deallocated after Kokkos::finalize" + details)
        self.assertEqual(result.returncode, expected_returncode, details)
        if expected_stdout is not None:
            self.assertIn(expected_stdout, result.stdout, details)
        if expected_stderr is not None:
            self.assertIn(expected_stderr, result.stderr, details)


class TestKokkosLifetime(ScenarioTestCase):

    def test_documented_reverse_order(self):
        """Control: objects released before the parser, as the documentation asks."""
        result = run_scenario("""
            kp = pycompadre.KokkosParser()
            gmls, helper = make_gmls()
            del helper
            del gmls
            del kp
            print("completed")
        """)
        self.assertScenario(result, expected_stdout="completed")

    def test_module_globals_at_exit(self):
        """Parser and objects as module-level globals, released by interpreter teardown."""
        result = run_scenario("""
            kp = pycompadre.KokkosParser()
            gmls, helper = make_gmls()
            print("completed")
        """)
        self.assertScenario(result, expected_stdout="completed")

    def test_parser_released_before_objects(self):
        """The parser is released while objects still own Views."""
        result = run_scenario("""
            kp = pycompadre.KokkosParser()
            gmls, helper = make_gmls()
            del kp
            del helper
            del gmls
            print("completed")
        """)
        self.assertScenario(result, expected_stdout="completed")

    def test_uncaught_exception_with_live_objects(self):
        """A script that fails must report its own error (exit code 1), not a Kokkos abort."""
        result = run_scenario("""
            kp = pycompadre.KokkosParser()
            gmls, helper = make_gmls()
            raise RuntimeError("intentional failure")
        """)
        self.assertScenario(result, expected_returncode=1, expected_stderr="intentional failure")

    def test_first_of_two_parsers_released(self):
        """Kokkos stays initialized while any parser is alive (as the KokkosParser docstring states)."""
        result = run_scenario("""
            kp1 = pycompadre.KokkosParser()
            kp2 = pycompadre.KokkosParser()
            del kp1
            assert pycompadre.Kokkos.is_initialized(), "Kokkos finalized while a KokkosParser is alive"
            gmls, helper = make_gmls()
            del helper
            del gmls
            del kp2
            print("completed")
        """)
        self.assertScenario(result, expected_stdout="completed")


class TestKokkosParserArguments(ScenarioTestCase):

    def test_initializes_from_orig_argv(self):
        """Run by the Python executable, KokkosParser() initializes Kokkos with sys.orig_argv."""
        result = run_scenario("""
            kp = pycompadre.KokkosParser()
            print("completed")
        """, command_line_args=["--kokkos-print-configuration"])
        # --kokkos-print-configuration, taken from the command line, makes Kokkos print its configuration
        self.assertScenario(result, expected_stdout="Kokkos Version")
        self.assertIn("completed", result.stdout)

    def test_command_line_lists_accepted(self):
        """sys.argv and sys.orig_argv (the same lists or equal ones) are accepted; later parsers join."""
        result = run_scenario("""
            parsers = [pycompadre.KokkosParser(sys.argv), pycompadre.KokkosParser(list(sys.argv)),
                       pycompadre.KokkosParser(sys.orig_argv), pycompadre.KokkosParser(list(sys.orig_argv)),
                       pycompadre.KokkosParser(None), pycompadre.KokkosParser()]
            gmls, helper = make_gmls()
            del helper, gmls, parsers
            print("completed")
        """, command_line_args=["--kokkos-disable-warnings"])
        self.assertScenario(result, expected_stdout="completed")

    def test_other_list_raises(self):
        """Any other list raises, since it would be ignored, and Kokkos is not initialized from it."""
        result = run_scenario("""
            try:
                pycompadre.KokkosParser(["program", "--kokkos-disable-warnings"])
                raise AssertionError("a list other than sys.argv or sys.orig_argv must raise")
            except RuntimeError as error:
                assert "ignored" in str(error), str(error)
            assert not pycompadre.Kokkos.is_initialized(), "Kokkos was initialized from a rejected list"
            kp = pycompadre.KokkosParser()
            print("completed")
        """)
        self.assertScenario(result, expected_stdout="completed")

    def test_parser_after_finalize_raises(self):
        """Kokkos cannot be initialized again once finalized, so a later KokkosParser raises."""
        result = run_scenario("""
            kp = pycompadre.KokkosParser()
            del kp
            assert not pycompadre.Kokkos.is_initialized()
            try:
                pycompadre.KokkosParser()
                raise AssertionError("a KokkosParser after Kokkos is finalized must raise")
            except RuntimeError as error:
                assert "finalized" in str(error), str(error)
            print("completed")
        """)
        self.assertScenario(result, expected_stdout="completed")

    def test_joins_kokkos_initialized_elsewhere(self):
        """KokkosParser() joins a Kokkos it did not initialize, and does not finalize it."""
        result = run_scenario("""
            pycompadre.Kokkos.initialize(sys.argv)
            kp = pycompadre.KokkosParser()
            gmls, helper = make_gmls()
            del helper, gmls, kp
            assert pycompadre.Kokkos.is_initialized(), "KokkosParser finalized a Kokkos it did not initialize"
            pycompadre.Kokkos.finalize()
            print("completed")
        """)
        self.assertScenario(result, expected_stdout="completed")


if __name__ == "__main__":
    unittest.main()

"""Certification protocol in the paper's message order: the certifier installs the specification and evaluation, the workload
returns a signed proposal with its public key, the provider sends its components encrypted under that key, and the
workload returns a certificate with id = H(components) and m = (FN, FP)."""

from .protocol import Provider, Workload, make_spec, verify_certificate

__all__ = ["Provider", "Workload", "make_spec", "verify_certificate"]


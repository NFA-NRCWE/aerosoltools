"""Internal, topic-grouped implementation modules for the core
Aerosol classes.

Each module defines a *mixin* whose methods are composed into the
public :class:`~aerosoltools.Aerosol1D` / :class:`Aerosol2D` /
:class:`AerosolAlt` classes. Splitting the (large) method bodies out
by topic keeps those classes readable while leaving their public API
unchanged. These modules are not part of the public API.
"""

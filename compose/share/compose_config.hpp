// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_COMPOSE_CONFIG_HPP
#define INCLUDE_COMPOSE_CONFIG_HPP

#ifdef COMPOSE_CONFIG_IS_CMAKE
# include "compose_config.h"
#else
// Purposely error out.
"A non-cmake build of scream is currently not supported."
#endif

#endif // INCLUDE_COMPOSE_CONFIG_HPP

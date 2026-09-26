\name{sfgpu_backends}
\alias{sfgpu_backends}
\title{Inspect Compiled and Available Backends}
\usage{
sfgpu_backends()
}
\value{
A named list with entries \code{cpu} and \code{cuda}. Each entry is a list
containing logical scalar fields \code{compiled} and \code{available}, and
character scalar fields \code{device} and \code{reason}. The character fields
are \code{NA_character_} when not applicable. \code{available} reports runtime
availability; a CUDA build can be compiled but have no usable device.
}
\description{
Reports whether each backend was compiled into the installed package and
whether it is available at runtime. A CPU-only build does not link to the CUDA
runtime. This function reports capability; it does not select a backend for
\code{sfgpu_distance()}.
}
\seealso{\code{sfgpu_distance}}

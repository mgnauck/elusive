# bypass - elusive

A quick 4kb intro for Deadline 2023 in JavaScript and WebGPU. We did not want to show up empty handed at the party after roughly 17 years of abscence from the demoscene.

supah - audio shader

warp - visual shader, intro code

## Run info

Final release can be found in the /release directory. You will need a WebGPU-capable browser, like Chromium/Chrome (version 113 or higher), Firefox Nightly or Edge. Please use the `--allow-file-access-from-files` option for Chrome or start a local webserver (`python3 -m http.server`) in the intro directory to run. Linux users might try their luck with `google-chrome-unstable --enable-unsafe-webgpu --use-vulkan=true --test-type`.

## Build info

You will need the following tools to package and compress the input files (JavaScript and shader).

https://github.com/mgnauck/wgslminify

https://github.com/mgnauck/js-payload-compress

https://github.com/terser/terser

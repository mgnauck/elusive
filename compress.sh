#!/bin/bash

# Script requires the following tools:
#
# https://github.com/mgnauck/wgslminify
# https://github.com/mgnauck/js-payload-compress
#
# Run: ./compress.sh player.js C,V,F

if [ ! -e wgslminify ] || [ ! -e js-payload-compress ]; then
  echo "Compression script needs wgslminify and js-payload-compress in current path!"
  exit 1
fi

infile=$1
infile_ext="${infile##*.}"
infile_name="${infile%.*}"
output_dir=output
shader_excludes=$2

rm -rf $output_dir
mkdir $output_dir

./wgslminify -e $shader_excludes audio.wgsl > $output_dir/audio_minified.wgsl
./wgslminify -e $shader_excludes visual.wgsl > $output_dir/visual_minified.wgsl

cd $output_dir

sed -f ../compress.sed ../$infile > ${infile_name}_with_shaders.${infile_ext}

terser ${infile_name}_with_shaders.${infile_ext} -m -c toplevel,passes=5,drop_console=true,unsafe=true,pure_getters=true,keep_fargs=false,booleans_as_integers=true --toplevel > ${infile_name}_minified.${infile_ext}

../js-payload-compress --zopfli-iterations=100 ${infile_name}_minified.${infile_ext} ${infile_name}_compressed.html

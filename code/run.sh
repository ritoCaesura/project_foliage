#!/bin/bash

for i in {1..50}
do
	c:/program\ files/blender\ foundation/blender-2.77/blender --background --python c:/blender/draft_main/basics100117.py -- $i
	# python c:/blender/draft_main/ground_tester.py -- $i
	#./blender -b -E CYCLES --python scripts/basics231216_02.py -- $i	
	python c:/blender/draft_main/pipeline.py -- $i
done
exec $SHELL

cd jni
call ndk-build
if %ERRORLEVEL% EQU 0 (
	echo ndk-build has failed, build cancelled
	cd..

	mkdir "assets\shaders\base"
	xcopy "..\..\data\shaders\base\*.spv" "assets\shaders\base" /Y
	

	mkdir "assets\shaders"
	xcopy "..\..\data\shaders\gears.vert.spv" "assets\shaders" /Y
	xcopy "..\..\data\shaders\gears.frag.spv" "assets\shaders" /Y
	
	mkdir "res\drawable"
	xcopy "..\..\android\images\icon.png" "res\drawable" /Y

	call ant debug -Dout.final.file=vulkanGears.apk
) ELSE (
	echo error : ndk-build failed with errors!
	cd..
)
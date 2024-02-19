@echo off
setlocal

rem Set LibTorch download URL
set LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.2.0%2Bcpu.zip
set LIBTORCH_ZIP=libtorch.zip

rem Create a directory for LibTorch if it doesn't exist
if not exist libtorch mkdir libtorch

rem Download LibTorch only if it hasn't been downloaded yet
if not exist libtorch\%LIBTORCH_ZIP% (
    echo Downloading LibTorch...
    powershell -Command "Invoke-WebRequest -Uri %LIBTORCH_URL% -OutFile libtorch\%LIBTORCH_ZIP%"
    echo Extracting LibTorch...
    powershell -Command "Expand-Archive -Path libtorch\%LIBTORCH_ZIP% -DestinationPath . -Force"
)

rem Create a build directory
if not exist build mkdir build
cd build

rem Run CMake to configure the project. Adjust the path to your LibTorch cmake folder
cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DTorch_DIR=../libtorch/share/cmake/Torch ..

cd ..

rem Copy compile_commands.json for IntelliSense (optional)
if exist build\compile_commands.json copy /y build\compile_commands.json .

echo Setup completed.

endlocal

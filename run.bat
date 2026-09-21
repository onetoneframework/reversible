@echo off
REM Reversible integer training - one entry point for every experiment here.
REM
REM   run.bat gates       verification gates          ~2 min
REM   run.bat memory      depth-independent memory    ~1 min
REM   run.bat demo        short train + translate     ~5 min
REM   run.bat train       overfit 512 Tatoeba pairs   ~16 min
REM   run.bat train-ted   real training on TED2020    ~32 min
REM   run.bat translate   translate with saved model  instant
REM   run.bat translate "<japanese text>"        translate your own text
REM   run.bat digits      original digits gate matrix ~3 min
REM   run.bat fashion     Fashion-MNIST conv/attn     ~4 min
REM   run.bat all         gates + memory + demo
setlocal

cd /d "%~dp0"

REM The `python` on PATH here is a broken MinGW stub, so find a real one.
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY (
  for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
)
if not defined PY (
  echo [error] No working Python found.
  echo         Expected %LOCALAPPDATA%\Programs\Python\Python313\python.exe
  exit /b 1
)

REM Japanese/Korean output needs UTF-8; this console defaults to CP949.
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul 2>&1

REM Single process, threads pinned. Oversubscribing has cost more than it bought.
if not defined THREADS set "THREADS=4"
set "OMP_NUM_THREADS=%THREADS%"
set "MKL_NUM_THREADS=%THREADS%"

set "CMD=%~1"
if "%CMD%"=="" set "CMD=help"

if /i "%CMD%"=="help"      goto :help
if /i "%CMD%"=="gates"     goto :gates
if /i "%CMD%"=="memory"    goto :memory
if /i "%CMD%"=="demo"      goto :demo
if /i "%CMD%"=="train"     goto :train
if /i "%CMD%"=="train-ted" goto :trainted
if /i "%CMD%"=="translate" goto :translate
if /i "%CMD%"=="digits"    goto :digits
if /i "%CMD%"=="fashion"   goto :fashion
if /i "%CMD%"=="all"       goto :all
echo Unknown command "%CMD%".
goto :help

:help
echo.
echo   run.bat gates       verification gates ^(inverse, cache-free, per-layer, attention^)
echo   run.bat memory      depth-independent activation memory, K = 2/4/8
echo   run.bat demo        short training then translate  ^(~5 min^)
echo   run.bat train       overfit 512 real Tatoeba pairs ^(~16 min^)
echo   run.bat train-ted   real training on TED2020       ^(~32 min^)
echo   run.bat translate   translate with the saved model
echo   run.bat digits      original sklearn-digits gate matrix
echo   run.bat fashion     Fashion-MNIST conv / attention
echo   run.bat all         gates + memory + demo
echo.
echo   Python: %PY%
echo   Threads: %THREADS%   ^(set THREADS=N to change^)
goto :eof

:gates
echo === Gate 1: corpus + tokenizer round-trip ===
"%PY%" reversible_jako_data.py --corpus tatoeba --vocab-size 2000 --max-len 32 --report
echo.
echo === Gates 2-5: inverse, cache-free==cached, per-layer==batched, attention cosine ===
"%PY%" reversible_jako_verify.py --all --threads %THREADS%
goto :eof

:memory
echo === Depth-independent activation memory ^(K = 2, 4, 8^) ===
"%PY%" reversible_jako_mt.py --mode memory --corpus tatoeba --vocab-size 2000 ^
  --dim 64 --ffn-hidden 128 --batch 16 --depths 2 4 8 --threads %THREADS%
goto :eof

:demo
echo === Short training on real Tatoeba JA-KO pairs, then translate ===
"%PY%" reversible_jako_mt.py --mode train --corpus tatoeba --max-pairs 512 ^
  --epochs 15 --batch 16 --dim 64 --ffn-hidden 128 --k-enc 2 --k-dec 3 ^
  --vocab-size 2000 --hidden-bits 16 --lr-bits 0 --show-samples 5 ^
  --eval-every 5 --threads %THREADS% --out jako_demo.json --save-ckpt jako_model.pt
echo.
echo === Translating with the model just trained ===
"%PY%" reversible_jako_mt.py --mode translate --ckpt jako_model.pt --threads %THREADS%
goto :eof

:train
echo === Gate 6: intentional overfit, 512 real Tatoeba pairs, 70 epochs ===
"%PY%" reversible_jako_mt.py --mode train --corpus tatoeba --max-pairs 512 ^
  --epochs 70 --batch 16 --dim 64 --ffn-hidden 128 --k-enc 2 --k-dec 3 ^
  --vocab-size 2000 --hidden-bits 16 --lr-bits 0 --show-samples 10 ^
  --eval-every 10 --threads %THREADS% --out jako_gate6_overfit.json --save-ckpt jako_model.pt
goto :eof

:trainted
echo === Gate 7: real training on TED2020, 6000 pairs ===
"%PY%" reversible_jako_mt.py --mode train --corpus ted2020 --max-pairs 6000 ^
  --epochs 12 --batch 16 --dim 64 --ffn-hidden 128 --k-enc 2 --k-dec 3 ^
  --vocab-size 3000 --hidden-bits 16 --lr-bits 0 --show-samples 10 ^
  --eval-every 3 --threads %THREADS% --out jako_results.json --save-ckpt jako_model.pt
goto :eof

:translate
if not exist jako_model.pt (
  echo [error] jako_model.pt not found. Run "run.bat demo" or "run.bat train" first.
  exit /b 1
)
if "%~2"=="" (
  "%PY%" reversible_jako_mt.py --mode translate --ckpt jako_model.pt --threads %THREADS%
) else (
  "%PY%" reversible_jako_mt.py --mode translate --ckpt jako_model.pt --threads %THREADS% --text %2 %3 %4 %5 %6 %7 %8 %9
)
goto :eof

:digits
echo === Original digits experiments: bit-exactness fingerprint and option gates ===
"%PY%" reversible_integer_golden_capture.py reversible_integer_golden_fingerprint.json
"%PY%" reversible_integer_gate_matrix.py
"%PY%" reversible_integer_structural.py
goto :eof

:fashion
echo === Fashion-MNIST reversible conv and attention ===
"%PY%" reversible_arch.py --arch conv --mode verify --depths 2 4 8 --threads %THREADS%
"%PY%" reversible_arch.py --arch attn --mode verify --depths 2 4 8 --threads %THREADS%
"%PY%" reversible_arch.py --arch conv --mode memory --depths 2 4 8 16 --threads %THREADS%
"%PY%" reversible_arch.py --arch attn --mode memory --depths 2 4 8 16 --threads %THREADS%
goto :eof

:all
call "%~f0" gates
call "%~f0" memory
call "%~f0" demo
goto :eof

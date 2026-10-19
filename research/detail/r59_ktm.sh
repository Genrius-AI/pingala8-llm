#!/bin/bash
# r59_ktm: ik IQ2_KT size-matched to B3 (same DP procedure as r56, budget = B3 bytes; research/detail/r59_ktm_dp.py),
# WikiText-2 ppl (r36_hfj protocol, ik binary), paired vs B3. Temp quant deleted after use (sha256 logged). Plain ASCII.
cd ~/repo; D=research/detail; LOG=logs/r59_ktm.log; W=work/r59kt; mkdir -p $W
IKR=ext/ik_llama.cpp; IK=$IKR/build-cuda/bin; F16=models/qwen3-4B-f16.gguf; IM=$W/qwen3-4B.imatrix.dat; KTM=$W/qwen3-4B-IQ2_KT-b3m-ik.gguf
B3B=1245915872; PRED=1245914752
CQ='blk\.([7-9]|1[0-9]|2[0-9])\.attn_output\.weight=iq2_kt,blk\.7\.attn_v\.weight=iq2_kt,blk\.([89]|1[0-9]|2[0-6])\.attn_v\.weight=iq3_k,blk\.(10|11)\.ffn_down\.weight=iq2_kt'
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }
say "r59_ktm start sid $(ps -o sid= -p $$ | tr -d ' ')"
if [ "$(fsize $KTM)" != "$PRED" ]; then
  rm -f $KTM; echo "$IK/llama-quantize --imatrix $IM --custom-q $CQ $F16 $KTM IQ2_KT 12" > $W/quant.cmd
  CPU=1 bin/job bash -c "source bin/cuda_env.sh; [ -s $IM ] || python -I $IKR/convert_imatrix_gguf_to_dat.py --outfile $IM models/qwen3-4B.imatrix > $W/imatrix_convert.log 2>&1; $IK/llama-quantize --imatrix $IM --custom-q '$CQ' $F16 $KTM IQ2_KT 12 > $W/quant.log 2>&1"
  say "QUANT rc=$? bytes=$(fsize $KTM) predicted=$PRED B3=$B3B types: $(grep -aoE 'type +iq[0-9a-z_]+: +[0-9]+' $W/quant.log | tr -s ' ' | tr '\n' ',')"
fi
[ "$(fsize $KTM)" -gt 0 ] && [ "$(fsize $KTM)" -le $B3B ] || { say "BLOCKER KTM bytes $(fsize $KTM) not in (0, $B3B]"; exit 5; }
sha256sum $KTM | cut -c1-64 > $W/ktm.sha256; say "sha256 $(cat $W/ktm.sha256)"
grep -aqE '\[73\][0-9]' logs/r36_hfj_r59_ktm.log 2>/dev/null || bin/job env HFJ_BIN=$IK bash $D/r36_hfj.sh r59_ktm $KTM >> $W/ppl_drive.log 2>&1
say "PPL KTM $(grep -a -o '\[73\][0-9.]*' logs/r36_hfj_r59_ktm.log | tail -1)"
say "PAIRED B3 vs KT-b3match: $(python3 $D/r59_paired.py logs/r36_hfj_r59_b3_native.log logs/r36_hfj_r59_ktm.log 2>&1 | tail -1)"
grep -aqE '\[73\][0-9]' logs/r36_hfj_r59_ktm.log && rm -f $KTM $IM
say "r59_ktm end; left in $W: $(ls $W | tr '\n' ' ')"

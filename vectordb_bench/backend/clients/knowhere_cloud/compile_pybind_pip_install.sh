rm -rf build \
&& mkdir build \
&& cd build \
&& conan install .. --build=missing -o with_diskann=True -s compiler.libcxx=libstdc++11 -s build_type=Release \
&& conan build .. \
&& cd .. \
&& cd python \
&& rm -rf dist \
&& python3 setup.py bdist_wheel \
&& pip3 install --force-reinstall dist/*.whl \
&& cd ..
